"""
Unit tests for mixed verify+extend flashinfer backend.

Verifies that the flashinfer attention backend can handle a batch containing
both verify tokens (spec decode, tree attention mask) and extend tokens
(chunk prefill, causal attention) by dispatching to two separate wrappers.

Tests:
1. ForwardBatch / PrefillMetadata field existence and defaults
2. init_forward_metadata routing and batch splitting
3. _forward_mixed_verify_extend tensor splitting and concatenation
4. No regression when num_verify_reqs == 0

Usage:
    python -m pytest test/registered/attention/test_mixed_verify_extend_backend.py -v
    python test/registered/attention/test_mixed_verify_extend_backend.py -v
"""

import dataclasses
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    PrefillMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_backend_mock():
    backend = MagicMock(spec=FlashInferAttnBackend)
    backend.prefill_wrappers_verify = [MagicMock(name="verify_wrapper")]
    backend.prefill_wrappers_paged = [MagicMock(name="paged_wrapper")]
    backend.is_multimodal = False
    backend.multi_item_scoring_delimiter = None
    backend.enable_deterministic = False
    backend.prefill_split_tile_size = None
    return backend


def _make_layer(tp_q_head_num=4, head_dim=64):
    layer = MagicMock()
    layer.tp_q_head_num = tp_q_head_num
    layer.head_dim = head_dim
    layer.v_head_dim = head_dim
    layer.is_cross_attention = False
    layer.scaling = 1.0 / (head_dim**0.5)
    layer.logit_cap = 0.0
    layer.sliding_window_size = -1
    layer.k_scale = 1.0
    layer.v_scale = 1.0
    layer.k_scale_float = 1.0
    layer.v_scale_float = 1.0
    layer.layer_id = 0
    return layer


def _make_forward_backend_mock(
    nv, ne, heads=2, dim=32, device="cuda"
):
    """Build mocked backend + wrappers + forward_batch for _forward_mixed_verify_extend."""
    verify_wrapper = MagicMock()
    extend_wrapper = MagicMock()

    meta = PrefillMetadata(
        [extend_wrapper],
        False,
        False,
        verify_wrappers=[verify_wrapper],
        num_verify_tokens=nv,
    )
    backend = MagicMock(spec=FlashInferAttnBackend)
    backend.forward_metadata = meta
    backend.num_wrappers = 1
    backend.dispatch_reason = None
    backend._get_wrapper_idx = MagicMock(return_value=0)

    total = nv + ne
    fb = MagicMock()
    fb.out_cache_loc = torch.arange(total, device=device)
    fb.token_to_kv_pool = MagicMock()
    fb.token_to_kv_pool.get_kv_buffer.return_value = MagicMock()

    return backend, verify_wrapper, extend_wrapper, fb


# ---------------------------------------------------------------------------
# 1. Field existence and defaults
# ---------------------------------------------------------------------------
class TestForwardBatchFields(CustomTestCase):
    def test_new_fields_exist(self):
        fields = {f.name: f for f in dataclasses.fields(ForwardBatch)}
        self.assertIn("num_verify_reqs", fields)
        self.assertIn("num_verify_tokens", fields)
        self.assertIn("verify_spec_info", fields)

    def test_default_values(self):
        fields = {f.name: f for f in dataclasses.fields(ForwardBatch)}
        self.assertEqual(fields["num_verify_reqs"].default, 0)
        self.assertEqual(fields["num_verify_tokens"].default, 0)
        self.assertIsNone(fields["verify_spec_info"].default)


class TestPrefillMetadataFields(CustomTestCase):
    def test_default_values(self):
        meta = PrefillMetadata([MagicMock()], use_ragged=False, extend_no_prefix=False)
        self.assertIsNone(meta.verify_wrappers)
        self.assertEqual(meta.num_verify_tokens, 0)

    def test_with_verify(self):
        vw = [MagicMock()]
        meta = PrefillMetadata(
            [MagicMock()],
            use_ragged=False,
            extend_no_prefix=False,
            verify_wrappers=vw,
            num_verify_tokens=10,
        )
        self.assertIs(meta.verify_wrappers, vw)
        self.assertEqual(meta.num_verify_tokens, 10)


# ---------------------------------------------------------------------------
# 2. init_forward_metadata routing
# ---------------------------------------------------------------------------
class TestInitForwardMetadataRouting(CustomTestCase):

    def test_mixed_verify_calls_updater_twice(self):
        """num_verify_reqs > 0 → indices_updater_prefill.update called twice."""
        backend = _make_backend_mock()
        nv_reqs, ne_reqs = 3, 5
        total = nv_reqs + ne_reqs

        update_calls = []

        def track_update(*args, **kwargs):
            prefix_lens = args[4] if len(args) > 4 else kwargs.get("prefix_lens")
            update_calls.append(
                {
                    "n_reqs": len(args[0]),
                    "n_seq_lens": len(args[1]),
                    "prefix_lens": prefix_lens,
                    "prefill_wrappers": kwargs.get("prefill_wrappers"),
                    "spec_info": kwargs.get("spec_info"),
                    "use_ragged": kwargs.get("use_ragged"),
                }
            )

        backend.indices_updater_prefill = MagicMock()
        backend.indices_updater_prefill.update = track_update

        fb = MagicMock()
        fb.forward_mode = ForwardMode.MIXED
        fb.num_verify_reqs = nv_reqs
        fb.num_verify_tokens = 15
        fb.verify_spec_info = MagicMock(name="verify_spec_info")
        fb.req_pool_indices = torch.arange(total)
        fb.seq_lens = torch.ones(total, dtype=torch.int32) * 100
        fb.seq_lens_cpu = torch.ones(total, dtype=torch.int32) * 100
        fb.seq_lens_sum = total * 100
        fb.extend_prefix_lens = torch.ones(ne_reqs, dtype=torch.int32) * 50
        fb.encoder_lens = None

        FlashInferAttnBackend.init_forward_metadata(backend, fb)

        self.assertEqual(len(update_calls), 2, "Should call update twice")

        # First call → verify subset
        c0 = update_calls[0]
        self.assertEqual(c0["n_reqs"], nv_reqs)
        self.assertEqual(c0["n_seq_lens"], nv_reqs)
        self.assertIsNone(c0["prefix_lens"])
        self.assertIs(c0["prefill_wrappers"], backend.prefill_wrappers_verify)
        self.assertIsNotNone(c0["spec_info"])
        self.assertFalse(c0["use_ragged"])

        # Second call → extend subset
        c1 = update_calls[1]
        self.assertEqual(c1["n_reqs"], ne_reqs)
        self.assertEqual(c1["n_seq_lens"], ne_reqs)
        self.assertIsNotNone(c1["prefix_lens"])
        self.assertIs(c1["prefill_wrappers"], backend.prefill_wrappers_paged)
        self.assertIsNone(c1["spec_info"])
        self.assertFalse(c1["use_ragged"])

        # forward_metadata carries verify info
        meta = backend.forward_metadata
        self.assertIsInstance(meta, PrefillMetadata)
        self.assertIs(meta.verify_wrappers, backend.prefill_wrappers_verify)
        self.assertEqual(meta.num_verify_tokens, 15)
        self.assertFalse(meta.use_ragged)

    def test_req_pool_indices_split_correctly(self):
        """req_pool_indices[:nv] goes to verify, [nv:] goes to extend."""
        backend = _make_backend_mock()
        nv_reqs, ne_reqs = 2, 4
        total = nv_reqs + ne_reqs

        captured_indices = []

        def track_update(*args, **kwargs):
            captured_indices.append(args[0].tolist())

        backend.indices_updater_prefill = MagicMock()
        backend.indices_updater_prefill.update = track_update

        fb = MagicMock()
        fb.forward_mode = ForwardMode.MIXED
        fb.num_verify_reqs = nv_reqs
        fb.num_verify_tokens = 8
        fb.verify_spec_info = MagicMock()
        fb.req_pool_indices = torch.tensor([10, 20, 30, 40, 50, 60])
        fb.seq_lens = torch.ones(total, dtype=torch.int32) * 100
        fb.seq_lens_cpu = torch.ones(total, dtype=torch.int32) * 100
        fb.seq_lens_sum = total * 100
        fb.extend_prefix_lens = torch.ones(ne_reqs, dtype=torch.int32) * 50
        fb.encoder_lens = None

        FlashInferAttnBackend.init_forward_metadata(backend, fb)

        self.assertEqual(captured_indices[0], [10, 20])
        self.assertEqual(captured_indices[1], [30, 40, 50, 60])

    @patch(
        "sglang.srt.layers.attention.flashinfer_backend.is_in_piecewise_cuda_graph",
        return_value=False,
    )
    def test_no_regression_without_verify(self, _mock_pcg):
        """num_verify_reqs == 0 → original extend path, update called once."""
        backend = _make_backend_mock()

        update_calls = []

        def track_update(*args, **kwargs):
            update_calls.append(kwargs)

        backend.indices_updater_prefill = MagicMock()
        backend.indices_updater_prefill.update = track_update

        fb = MagicMock()
        fb.forward_mode = ForwardMode.MIXED
        fb.num_verify_reqs = 0
        fb.extend_prefix_lens = torch.ones(5, dtype=torch.int32) * 50
        fb.extend_prefix_lens_cpu = [50] * 5
        fb.req_pool_indices = torch.arange(5)
        fb.seq_lens = torch.ones(5, dtype=torch.int32) * 100
        fb.seq_lens_cpu = torch.ones(5, dtype=torch.int32) * 100
        fb.seq_lens_sum = 500
        fb.encoder_lens = None

        FlashInferAttnBackend.init_forward_metadata(backend, fb)

        self.assertEqual(len(update_calls), 1, "Should call update once (normal extend)")
        self.assertIs(
            update_calls[0]["prefill_wrappers"], backend.prefill_wrappers_paged
        )
        self.assertIsNone(update_calls[0]["spec_info"])

        meta = backend.forward_metadata
        self.assertIsNone(meta.verify_wrappers)
        self.assertEqual(meta.num_verify_tokens, 0)


# ---------------------------------------------------------------------------
# 3. _forward_mixed_verify_extend tensor logic (CUDA)
# ---------------------------------------------------------------------------
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestForwardMixedVerifyExtend(CustomTestCase):

    def test_output_shape_and_values(self):
        """q split → two wrappers → cat should preserve shape and values."""
        nv, ne = 5, 8
        total = nv + ne
        heads, dim = 4, 64
        hidden = heads * dim

        q = torch.randn(total, hidden, device="cuda")
        k = torch.randn(total, hidden, device="cuda")
        v = torch.randn(total, hidden, device="cuda")

        verify_out = torch.full((nv, heads, dim), 1.0, device="cuda")
        extend_out = torch.full((ne, heads, dim), 2.0, device="cuda")

        backend, vw, ew, fb = _make_forward_backend_mock(nv, ne, heads, dim)
        vw.forward.return_value = verify_out
        ew.forward.return_value = extend_out

        result = FlashInferAttnBackend._forward_mixed_verify_extend(
            backend, q, k, v, _make_layer(heads, dim), fb, save_kv_cache=True
        )

        self.assertEqual(result.shape, (total, hidden))
        r = result.view(total, heads, dim)
        self.assertTrue(torch.allclose(r[:nv], verify_out))
        self.assertTrue(torch.allclose(r[nv:], extend_out))

    def test_q_split_shapes(self):
        """verify wrapper receives q[:nv], extend wrapper receives q[nv:]."""
        nv, ne = 3, 7
        heads, dim = 2, 32
        total = nv + ne
        hidden = heads * dim

        q = torch.randn(total, hidden, device="cuda")
        k = torch.randn(total, hidden, device="cuda")
        v = torch.randn(total, hidden, device="cuda")

        backend, vw, ew, fb = _make_forward_backend_mock(nv, ne, heads, dim)
        vw.forward.return_value = torch.zeros(nv, heads, dim, device="cuda")
        ew.forward.return_value = torch.zeros(ne, heads, dim, device="cuda")

        FlashInferAttnBackend._forward_mixed_verify_extend(
            backend, q, k, v, _make_layer(heads, dim), fb, save_kv_cache=True
        )

        vq = vw.forward.call_args[0][0]
        eq = ew.forward.call_args[0][0]
        self.assertEqual(vq.shape, (nv, heads, dim))
        self.assertEqual(eq.shape, (ne, heads, dim))

    def test_kv_cache_written_before_attention(self):
        """set_kv_buffer must be called before either wrapper.forward."""
        nv, ne = 2, 3
        heads, dim = 2, 32
        total = nv + ne
        hidden = heads * dim

        q = torch.randn(total, hidden, device="cuda")
        k = torch.randn(total, hidden, device="cuda")
        v = torch.randn(total, hidden, device="cuda")

        call_order = []
        backend, vw, ew, fb = _make_forward_backend_mock(nv, ne, heads, dim)
        vw.forward.side_effect = lambda *a, **kw: (
            call_order.append("verify"),
            torch.zeros(nv, heads, dim, device="cuda"),
        )[1]
        ew.forward.side_effect = lambda *a, **kw: (
            call_order.append("extend"),
            torch.zeros(ne, heads, dim, device="cuda"),
        )[1]
        fb.token_to_kv_pool.set_kv_buffer.side_effect = (
            lambda *a, **kw: call_order.append("set_kv")
        )

        FlashInferAttnBackend._forward_mixed_verify_extend(
            backend, q, k, v, _make_layer(heads, dim), fb, save_kv_cache=True
        )

        self.assertEqual(call_order, ["set_kv", "verify", "extend"])

    def test_skip_kv_cache_when_save_false(self):
        """save_kv_cache=False → set_kv_buffer not called."""
        nv, ne = 2, 3
        heads, dim = 2, 32
        total = nv + ne
        hidden = heads * dim

        q = torch.randn(total, hidden, device="cuda")
        k = torch.randn(total, hidden, device="cuda")
        v = torch.randn(total, hidden, device="cuda")

        backend, vw, ew, fb = _make_forward_backend_mock(nv, ne, heads, dim)
        vw.forward.return_value = torch.zeros(nv, heads, dim, device="cuda")
        ew.forward.return_value = torch.zeros(ne, heads, dim, device="cuda")

        FlashInferAttnBackend._forward_mixed_verify_extend(
            backend, q, k, v, _make_layer(heads, dim), fb, save_kv_cache=False
        )

        fb.token_to_kv_pool.set_kv_buffer.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Dispatch condition
# ---------------------------------------------------------------------------
class TestForwardExtendDispatch(CustomTestCase):

    def test_dispatch_condition_true(self):
        meta = PrefillMetadata(
            [MagicMock()], False, False,
            verify_wrappers=[MagicMock()], num_verify_tokens=5,
        )
        self.assertIsNotNone(meta.verify_wrappers)
        self.assertGreater(meta.num_verify_tokens, 0)

    def test_dispatch_condition_false_no_wrappers(self):
        meta = PrefillMetadata([MagicMock()], False, False)
        cond = meta.verify_wrappers is not None and meta.num_verify_tokens > 0
        self.assertFalse(cond)

    def test_dispatch_condition_false_zero_tokens(self):
        meta = PrefillMetadata(
            [MagicMock()], False, False,
            verify_wrappers=[MagicMock()], num_verify_tokens=0,
        )
        cond = meta.verify_wrappers is not None and meta.num_verify_tokens > 0
        self.assertFalse(cond)


# ---------------------------------------------------------------------------
# 5. ForwardMode branch conditions
# ---------------------------------------------------------------------------
class TestForwardModeBranching(CustomTestCase):

    def test_mixed_is_mixed(self):
        self.assertTrue(ForwardMode.MIXED.is_mixed())

    def test_other_modes_not_mixed(self):
        for mode in [
            ForwardMode.EXTEND,
            ForwardMode.DECODE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND,
            ForwardMode.IDLE,
        ]:
            self.assertFalse(mode.is_mixed(), f"{mode} should not be mixed")

    def test_mixed_is_extend(self):
        self.assertTrue(ForwardMode.MIXED.is_extend())

    def test_mixed_not_decode(self):
        self.assertFalse(ForwardMode.MIXED.is_decode_or_idle())


if __name__ == "__main__":
    unittest.main(verbosity=2)
