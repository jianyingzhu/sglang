"""Reconcile SWA evictable_size_ counter vs tree / allocator (pool leak triage)."""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("SGLANG_SWA_EVICTABLE_DIAG", "1") != "0"
_by_tag: Dict[str, int] = defaultdict(int)


def enabled() -> bool:
    return _ENABLED


def bump(cache: "SWARadixCache", delta: int, tag: str, **ctx: Any) -> None:
    if not _ENABLED or delta == 0:
        return
    _by_tag[tag] += delta


def count_mapped_swa_slots(cache: "SWARadixCache", value) -> int:
    if value is None or len(value) == 0:
        return 0
    allocator = cache.token_to_kv_pool_allocator
    mapping = getattr(allocator, "full_to_swa_index_mapping", None)
    if mapping is None:
        return len(value)
    return int((mapping[value] > 0).sum().item())


def reconcile(cache: "SWARadixCache") -> Dict[str, Any]:
    """Compare counter / tree / allocator views of the SWA pool."""
    allocator = cache.token_to_kv_pool_allocator
    total = int(getattr(allocator, "swa_size", lambda: 0)() or 0)
    if hasattr(allocator, "swa_attn_allocator"):
        total = int(allocator.swa_attn_allocator.size)

    available = int(allocator.swa_available_size())
    counter_evictable = int(cache.swa_evictable_size())
    counter_protected = int(cache.swa_protected_size())

    tree_unlocked = 0
    tree_locked = 0
    tree_mapped_slots = 0
    heterogeneous: List[Dict[str, Any]] = []

    for node in cache._collect_nontombstone_nodes():
        slots = cache._swa_slots_in_value(node.value)
        mapped = count_mapped_swa_slots(cache, node.value)
        tree_mapped_slots += mapped
        if node.swa_lock_ref == 0:
            tree_unlocked += slots
        else:
            tree_locked += slots
        if mapped != slots:
            heterogeneous.append(
                {
                    "node_id": getattr(node, "id", None),
                    "len_value": slots,
                    "mapped": mapped,
                    "swa_lock_ref": node.swa_lock_ref,
                    "full_lock_ref": node.full_lock_ref,
                }
            )

    lru_unlocked = 0
    node = cache.swa_lru_list.get_lru_no_lock()
    while cache.swa_lru_list.in_list(node):
        lru_unlocked += len(node.value)
        node = cache.swa_lru_list.get_prev_no_lock(node)

    pool_accounting_gap = (
        available + counter_evictable + counter_protected - total
    )
    phantom_evictable = counter_evictable - tree_unlocked
    phantom_protected = counter_protected - tree_locked
    mapped_gap = tree_mapped_slots - (total - available)

    return {
        "total": total,
        "available": available,
        "counter_evictable": counter_evictable,
        "counter_protected": counter_protected,
        "tree_unlocked": tree_unlocked,
        "tree_locked": tree_locked,
        "tree_mapped_slots": tree_mapped_slots,
        "lru_unlocked": lru_unlocked,
        "phantom_evictable": phantom_evictable,
        "phantom_protected": phantom_protected,
        "pool_accounting_gap": pool_accounting_gap,
        "mapped_gap": mapped_gap,
        "heterogeneous_node_count": len(heterogeneous),
        "heterogeneous_phantom_slots": sum(
            h["len_value"] - h["mapped"] for h in heterogeneous
        ),
        "first_heterogeneous": heterogeneous[0] if heterogeneous else None,
        "heterogeneous_sample": heterogeneous[:5],
    }


def format_reconcile(snap: Dict[str, Any]) -> str:
    lines = [
        "[SWA-EVICT-DIAG] RECONCILE "
        f"total={snap['total']} available={snap['available']} "
        f"counter_evictable={snap['counter_evictable']} "
        f"counter_protected={snap['counter_protected']} "
        f"pool_gap={snap['pool_accounting_gap']}",
        "[SWA-EVICT-DIAG] RECONCILE tree_unlocked="
        f"{snap['tree_unlocked']} tree_locked={snap['tree_locked']} "
        f"tree_mapped={snap['tree_mapped_slots']} lru_unlocked={snap['lru_unlocked']}",
        "[SWA-EVICT-DIAG] RECONCILE phantom_evictable="
        f"{snap['phantom_evictable']} phantom_protected={snap['phantom_protected']} "
        f"mapped_gap={snap['mapped_gap']} het_nodes={snap['heterogeneous_node_count']} "
        f"het_phantom_slots={snap['heterogeneous_phantom_slots']}",
    ]
    if snap.get("heterogeneous_sample"):
        lines.append(
            "[SWA-EVICT-DIAG] RECONCILE het_sample="
            + repr(snap["heterogeneous_sample"])
        )
    if _by_tag:
        top = sorted(_by_tag.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
        lines.append(
            "[SWA-EVICT-DIAG] RECONCILE bump_by_tag="
            + ", ".join(f"{t}:{d:+d}" for t, d in top)
        )
    return "\n".join(lines)


def log_reconcile(cache: "SWARadixCache", reason: str = "") -> None:
    if not _ENABLED:
        return
    snap = reconcile(cache)
    msg = format_reconcile(snap)
    if reason:
        logger.warning("%s reason=%s", msg, reason)
    else:
        logger.info(msg)
