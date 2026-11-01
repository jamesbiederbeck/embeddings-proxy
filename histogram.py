"""Server-side port of photo-embedder's histogram_from_sims(): an adaptive-bin
histogram plus distributional stats over a 1-D array of cosine similarity
scores, used by /search's optional `histogram` field (TICKET-009). Ported
verbatim from photo_embedder/app.py:342-376 so this stays consistent with the
reference implementation's validated behavior, rather than being redesigned
from scratch.
"""

import math
from typing import Any


def _safe_float(x: float) -> float | None:
    # NaN isn't valid JSON (FastAPI's default encoder rejects it outright) — skew/kurtosis
    # go NaN when std is 0 (near-constant scores), which the reference implementation
    # never had to handle since real-world score distributions aren't perfectly constant.
    return None if math.isnan(x) else float(x)


def histogram_from_sims(sims: Any, buckets: int | None = None) -> dict:
    """`sims` should already exclude any self-match and be scoped to the
    query's intended population — this function has no scope/filter knowledge
    of its own; the caller must pass an already-correctly-scoped array.
    """
    import numpy as np
    from scipy import stats as spstats

    sims = np.asarray(sims, dtype=float)
    n = sims.size
    mean, std = float(sims.mean()), float(sims.std())
    lo, hi = float(sims.min()), float(sims.max())
    skew, kurt = _safe_float(spstats.skew(sims)), _safe_float(spstats.kurtosis(sims))

    # D'Agostino-Pearson omnibus test needs at least 8 samples; the reference
    # implementation assumes large corpora and doesn't guard this, but a small
    # scope (or a tight histogram_limit) can hand us far fewer points.
    if n >= 8:
        try:
            k2_raw, pval_raw = spstats.normaltest(sims)
            k2, pval = _safe_float(k2_raw), _safe_float(pval_raw)
            looks_normal = bool(pval is not None and pval > 0.05)
        except ValueError:
            k2 = pval = None
            looks_normal = False
    else:
        k2 = pval = None
        looks_normal = False

    if buckets is None:
        # Freedman-Diaconis bin width from the IQR, clamped to a sane count.
        q25, q75 = np.percentile(sims, [25, 75])
        iqr = q75 - q25
        bw = 2 * iqr * n ** (-1 / 3) if iqr > 0 else 0
        buckets = int(np.clip(round((hi - lo) / bw), 15, 60)) if bw > 0 else 30
    pad = (hi - lo) * 0.02 or 0.01
    edges = np.linspace(lo - pad, hi + pad, buckets + 1)
    counts, _ = np.histogram(sims, bins=edges)

    return {
        "n": int(n),
        "buckets_requested": buckets,
        "stats": {
            "mean": mean,
            "std": std,
            "median": float(np.median(sims)),
            "min": lo,
            "max": hi,
            "skewness": skew,
            "excess_kurtosis": kurt,
            "normaltest_statistic": k2,
            "normaltest_p": pval,
            "looks_normal": looks_normal,
        },
        "above_mean_plus_2std": int((sims > mean + 2 * std).sum()),
        "above_mean_plus_3std": int((sims > mean + 3 * std).sum()),
        "mean_plus_2std": mean + 2 * std,
        "buckets": [
            {
                "lo": round(float(edges[i]), 4),
                "hi": round(float(edges[i + 1]), 4),
                "count": int(counts[i]),
            }
            for i in range(buckets)
        ],
    }
