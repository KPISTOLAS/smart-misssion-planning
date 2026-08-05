"""Statistical export harness: mandatory CI, Wilcoxon, Cliff's delta, BH-FDR."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence
import numpy as np
from scipy import stats


@dataclass
class StatRow:
    name: str
    n_seeds: int
    mean: float
    std: float
    ci95_low: float
    ci95_high: float
    raw: List[float] | None = None


def ci95(samples: np.ndarray) -> tuple[float, float, float, float]:
    """Return mean, std, ci_low, ci_high (normal 95% CI)."""
    x = np.asarray(samples, dtype=float)
    n = len(x)
    m = float(np.mean(x)) if n else 0.0
    s = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = s / max(np.sqrt(n), 1.0)
    return m, s, m - 1.96 * se, m + 1.96 * se


def aggregate_metric(name: str, samples: Sequence[float]) -> StatRow:
    m, s, lo, hi = ci95(np.asarray(samples))
    return StatRow(name=name, n_seeds=len(samples), mean=m, std=s,
                   ci95_low=lo, ci95_high=hi, raw=list(samples))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size for paired or unpaired samples."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        # unpaired
        gt = sum(a > b for a in x for b in y)
        lt = sum(a < b for a in x for b in y)
        return float((gt - lt) / (len(x) * len(y)))
    gt = lt = 0
    for a, b in zip(x, y):
        if a > b:
            gt += 1
        elif a < b:
            lt += 1
    return float((gt - lt) / max(len(x), 1))


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank; returns (statistic, p_value)."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) != len(y):
        raise ValueError("Wilcoxon requires paired samples (same seeds).")
    if np.allclose(x, y):
        return 0.0, 1.0
    try:
        res = stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return 0.0, 1.0


def benjamini_hochberg(p_values: List[float]) -> List[float]:
    """BH-FDR adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(prev, p[order[i]] * n / rank)
        ranked[order[i]] = val
        prev = val
    return ranked.tolist()


@dataclass
class ComparisonRow:
    variant: str
    reference: str
    metric: str
    mean_diff: float
    wilcoxon_stat: float
    wilcoxon_p: float
    cliffs_delta: float
    fdr_p: float | None = None


def compare_variants(reference: StatRow, variants: Dict[str, StatRow],
                     metric_name: str = "metric") -> List[ComparisonRow]:
    """Pairwise Wilcoxon + Cliff's delta vs reference (same seed order required)."""
    rows = []
    pvals = []
    for name, var in variants.items():
        if reference.raw is None or var.raw is None:
            continue
        if len(reference.raw) != len(var.raw):
            raise ValueError(f"Seed pairing mismatch: {reference.name} vs {name}")
        w_stat, w_p = paired_wilcoxon(np.array(var.raw), np.array(reference.raw))
        cd = cliffs_delta(np.array(var.raw), np.array(reference.raw))
        rows.append(ComparisonRow(
            variant=name, reference=reference.name, metric=metric_name,
            mean_diff=var.mean - reference.mean,
            wilcoxon_stat=w_stat, wilcoxon_p=w_p, cliffs_delta=cd,
        ))
        pvals.append(w_p)
    fdr = benjamini_hochberg(pvals)
    for i, row in enumerate(rows):
        row.fdr_p = fdr[i]
    return rows


def stat_row_to_dict(row: StatRow) -> dict:
    d = asdict(row)
    d.pop("raw", None)  # omit raw from table export by default
    return d


def validate_table_rows(rows: List[StatRow], context: str = "table"):
    """Raise if any row lacks CI fields."""
    for r in rows:
        if r.n_seeds <= 0:
            raise ValueError(f"{context}: row {r.name} has n_seeds=0")
        if not np.isfinite(r.ci95_low) or not np.isfinite(r.ci95_high):
            raise ValueError(f"{context}: row {r.name} missing CI95")


def export_stat_table(rows: List[StatRow], comparisons: List[ComparisonRow] | None = None) -> dict:
    validate_table_rows(rows)
    out: Dict[str, Any] = {"rows": [stat_row_to_dict(r) for r in rows]}
    if comparisons:
        out["comparisons"] = [asdict(c) for c in comparisons]
    return out
