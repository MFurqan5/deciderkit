"""Pick thresholds from labeled data instead of by hand.

Input rows (JSONL): ``{"p": 0.91, "label": true}`` where ``label`` is the TRUTH: true means the tool call
really was fine (grounded) and should proceed. Rows from the audit log also work if you add a ``label``
field: the P(true) is read from the check named by ``--check``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TuneResult:
    n: int
    allow_above: float | None
    block_below: float | None
    auto_proceed_precision: float | None
    auto_block_precision: float | None
    auto_decided: float
    wrong_auto_proceeds: int
    wrong_auto_blocks: int
    note: str = ""


def rows_from_log(rows: list[dict[str, Any]], check: str | None = None) -> list[tuple[float, bool]]:
    out: list[tuple[float, bool]] = []
    for r in rows:
        label = r.get("label")
        per_check = r.get("labels")
        if check and isinstance(per_check, dict) and isinstance(per_check.get(check), bool):
            label = per_check[check]  # truth for THIS check (a call can be bad for a different reason)
        if not isinstance(label, bool):
            continue
        p = r.get("p")
        if p is None:
            for c in r.get("checks", []):
                if (check is None or c.get("name") == check) and isinstance(c.get("value"), (int, float)):
                    p = c["value"]
                    break
        if isinstance(p, (int, float)):
            out.append((float(p), label))
    return out


def tune(
    data: list[tuple[float, bool]],
    *,
    proceed_precision: float = 0.99,
    block_precision: float = 0.95,
    min_support: int = 5,
) -> TuneResult:
    """Choose allow_above / block_below so auto-decisions meet the precision targets.

    allow_above: lowest threshold where cases at or above it are truly fine >= ``proceed_precision``.
    block_below: highest threshold where cases at or below it are truly bad  >= ``block_precision``.
    Everything between goes to the ladder (guide, confirm or a bigger model).
    """
    n = len(data)
    if n == 0:
        return TuneResult(0, None, None, None, None, 0.0, 0, 0, "no labeled rows")
    cands = sorted({p for p, _ in data})

    allow: float | None = None
    for t in cands:
        sel = [ok for p, ok in data if p >= t]
        if len(sel) >= min_support and sum(sel) / len(sel) >= proceed_precision:
            allow = t
            break

    block: float | None = None
    for t in reversed(cands):
        sel = [ok for p, ok in data if p <= t]
        if len(sel) >= min_support and (len(sel) - sum(sel)) / len(sel) >= block_precision:
            block = t
            break

    note = ""
    if allow is not None and block is not None and block >= allow:
        note = "thresholds overlap; labels may be noisy. Nothing auto-decided safely."
        allow = block = None

    ap = [ok for p, ok in data if allow is not None and p >= allow]
    bl = [ok for p, ok in data if block is not None and p <= block]
    return TuneResult(
        n=n,
        allow_above=allow,
        block_below=block,
        auto_proceed_precision=(sum(ap) / len(ap)) if ap else None,
        auto_block_precision=((len(bl) - sum(bl)) / len(bl)) if bl else None,
        auto_decided=(len(ap) + len(bl)) / n,
        wrong_auto_proceeds=len(ap) - sum(ap),
        wrong_auto_blocks=sum(bl),
        note=note,
    )
