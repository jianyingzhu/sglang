"""Diagnostics for SWA inc_lock_ref / dec_lock_ref pairing (debug one-run triage)."""
from __future__ import annotations

import logging
import os
import traceback
from collections import defaultdict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default ON for active triage; set SGLANG_SWA_LOCK_DIAG=0 to disable.
_ENABLED = os.getenv("SGLANG_SWA_LOCK_DIAG", "1") != "0"

_inc_total = 0
_dec_total = 0
_dec_no_uuid = 0
_dec_skip_swa = 0
_dec_swa_only = 0
_by_tag: Dict[str, Dict[str, int]] = defaultdict(lambda: {"inc": 0, "dec": 0})


def enabled() -> bool:
    return _ENABLED


def _bump(tag: str, op: str) -> None:
    _by_tag[tag][op] += 1


def log_site(tag: str, op: str, **fields: Any) -> None:
    """Context log from call sites (does not bump running counters)."""
    if not _ENABLED:
        return
    parts = [f"[SWA-LOCK-DIAG] SITE {op.upper()} tag={tag}"]
    for k, v in fields.items():
        parts.append(f"{k}={v!r}")
    parts.append(
        f"running_inc={_inc_total} running_dec={_dec_total} dec_no_uuid={_dec_no_uuid}"
    )
    logger.info(" ".join(parts))


def log_inc_result(
    tag: str,
    leaf_node_id: Any,
    swa_uuid_for_lock: Optional[int],
    swa_lock_size: int,
    nodes_locked: int,
    evictable_before: int,
    evictable_after: int,
    protected_before: int,
    protected_after: int,
    **extra: Any,
) -> None:
    if not _ENABLED:
        return
    global _inc_total
    _inc_total += 1
    _bump(tag, "inc")
    logger.info(
        "[SWA-LOCK-DIAG] INC tag=%s leaf_node=%s swa_uuid=%s swa_lock_size=%d "
        "nodes_locked=%d evictable %d->%d (delta=%d) protected %d->%d "
        "running_inc=%d running_dec=%d %s",
        tag,
        leaf_node_id,
        swa_uuid_for_lock,
        swa_lock_size,
        nodes_locked,
        evictable_before,
        evictable_after,
        evictable_after - evictable_before,
        protected_before,
        protected_after,
        _inc_total,
        _dec_total,
        " ".join(f"{k}={v!r}" for k, v in extra.items()),
    )


def log_dec_result(
    tag: str,
    leaf_node_id: Any,
    swa_uuid_for_lock: Optional[int],
    skip_swa: bool,
    swa_steps: int,
    full_steps: int,
    evictable_before: int,
    evictable_after: int,
    protected_before: int,
    protected_after: int,
    hit_tombstone: bool = False,
    **extra: Any,
) -> None:
    if not _ENABLED:
        return
    global _dec_total, _dec_no_uuid, _dec_skip_swa
    _dec_total += 1
    _bump(tag, "dec")
    if swa_uuid_for_lock is None and not skip_swa:
        _dec_no_uuid += 1
    if skip_swa:
        _dec_skip_swa += 1
    level = logging.WARNING if (swa_uuid_for_lock is None and not skip_swa) else logging.INFO
    logger.log(
        level,
        "[SWA-LOCK-DIAG] DEC tag=%s leaf_node=%s swa_uuid=%s skip_swa=%s "
        "swa_steps=%d full_steps=%d hit_tombstone=%s "
        "evictable %d->%d (delta=%+d) protected %d->%d "
        "running_inc=%d running_dec=%d dec_no_uuid=%d %s",
        tag,
        leaf_node_id,
        swa_uuid_for_lock,
        skip_swa,
        swa_steps,
        full_steps,
        hit_tombstone,
        evictable_before,
        evictable_after,
        evictable_after - evictable_before,
        protected_before,
        protected_after,
        _inc_total,
        _dec_total,
        _dec_no_uuid,
        " ".join(f"{k}={v!r}" for k, v in extra.items()),
    )


def log_dec_swa_only(
    tag: str,
    leaf_node_id: Any,
    swa_uuid_for_lock: Optional[int],
    swa_steps: int,
    evictable_before: int,
    evictable_after: int,
    **extra: Any,
) -> None:
    if not _ENABLED:
        return
    global _dec_swa_only, _dec_total
    _dec_swa_only += 1
    _dec_total += 1
    _bump(tag, "dec")
    logger.info(
        "[SWA-LOCK-DIAG] DEC_SWA_ONLY tag=%s leaf_node=%s swa_uuid=%s swa_steps=%d "
        "evictable %d->%d (delta=%+d) running_dec_swa_only=%d %s",
        tag,
        leaf_node_id,
        swa_uuid_for_lock,
        swa_steps,
        evictable_before,
        evictable_after,
        evictable_after - evictable_before,
        _dec_swa_only,
        " ".join(f"{k}={v!r}" for k, v in extra.items()),
    )


def log_assert_near_miss(
    where: str,
    node_id: Any,
    full_lock_ref: int,
    swa_lock_ref: int,
    swa_tombstone: bool,
    **extra: Any,
) -> None:
    if not _ENABLED:
        return
    logger.error(
        "[SWA-LOCK-DIAG] NEAR_MISS where=%s node=%s full_lock_ref=%d swa_lock_ref=%d "
        "swa_tombstone=%s %s\n%s",
        where,
        node_id,
        full_lock_ref,
        swa_lock_ref,
        swa_tombstone,
        " ".join(f"{k}={v!r}" for k, v in extra.items()),
        "".join(traceback.format_stack(limit=8)[:-1]),
    )


def format_summary() -> str:
    imbalance = _inc_total - _dec_total
    lines = [
        f"[SWA-LOCK-DIAG] SUMMARY inc_total={_inc_total} dec_total={_dec_total} "
        f"imbalance(inc-dec)={imbalance} dec_no_uuid={_dec_no_uuid} "
        f"dec_skip_swa={_dec_skip_swa} dec_swa_only={_dec_swa_only}",
    ]
    for tag, counts in sorted(_by_tag.items()):
        lines.append(
            f"[SWA-LOCK-DIAG] SUMMARY tag={tag} inc={counts['inc']} dec={counts['dec']} "
            f"delta={counts['inc'] - counts['dec']}"
        )
    return "\n".join(lines)


def log_summary(reason: str = "") -> None:
    if not _ENABLED:
        return
    msg = format_summary()
    if reason:
        logger.warning("%s reason=%s", msg, reason)
    else:
        logger.info(msg)
