"""Phase 8: permutation tests, moving-block bootstrap and Holm correction (§15.8, B.4)."""
import numpy as np

from lcbank.analysis.data import DatasetData, History
from lcbank.analysis.evaluate import RHOS, SEEDS, as_curve, get_scorer
from lcbank.analysis.methods import METHODS, TIMING_METHODS


def permutation_p(observed, null_values, one_sided=True):
    """p = (1 + #{null >= observed}) / (1 + #perm); two-sided compares |values| (§15.8)."""
    null = np.asarray(null_values, dtype=float)
    obs = observed if one_sided else abs(observed)
    cmp = null if one_sided else np.abs(null)
    return float((1 + np.sum(cmp >= obs)) / (1 + len(null)))


def holm(pvalues):
    """Holm-corrected p-values, order preserved."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return {k: out[k] for k in pvalues}


def _block_resample_stats(values, block, resamples, seed, statistic):
    """Moving-block resamples of a series over test sittings (§15.8): block length 2, 2,000 resamples."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = max(1, n - block + 1)
    stats = []
    for _ in range(resamples):
        starts = rng.integers(0, starts_max, size=n_blocks)
        sample = np.concatenate([v[s:s + block] for s in starts])[:n]
        stats.append(statistic(sample))
    return v, n_blocks, np.asarray(stats, dtype=float)


def block_interval(values, level=0.95, block=2, resamples=2000, seed=SEEDS["bootstrap"], statistic=np.mean,
                   method="t_blocks"):
    """CI of a statistic over test sittings from moving-block resamples.

    `percentile` is what the locked plan specified; the simulator showed it under-covers badly when a dataset has
    few test sittings — with 8 sittings (PM) a nominal 95% interval covered the truth 84.8%, failing the §15.7
    control. The resampling is unchanged; `t_blocks` instead takes the bootstrap standard error and the t quantile
    with df = number of blocks - 1, which is the quantity the small number of blocks actually supports. See
    EX-008 / D-031: this is a post-lock change, so every interval it produces is exploratory (R7).
    """
    v, n_blocks, stats = _block_resample_stats(values, block, resamples, seed, statistic)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    tail = (1 - level) / 2 * 100
    if method == "percentile":
        lo, hi = np.percentile(stats, [tail, 100 - tail])
    elif method == "t_blocks":
        from scipy import stats as sps
        theta, se = float(statistic(v)), float(stats.std(ddof=1))
        q = float(sps.t.ppf(1 - (1 - level) / 2, max(n_blocks - 1, 1)))
        lo, hi = theta - q * se, theta + q * se
    else:
        raise ValueError(f"unknown interval method {method!r}")
    return float(lo), float(hi)


def moving_block_bootstrap(values, block=2, resamples=2000, seed=SEEDS["bootstrap"], statistic=np.mean,
                           method="t_blocks"):
    """95% CI over test sittings with a moving block bootstrap (§15.8, interval per EX-008)."""
    return block_interval(values, 0.95, block, resamples, seed, statistic, method)


def negligible(values, block=2, resamples=2000, seed=SEEDS["bootstrap"], margin=0.05, method="t_blocks"):
    """True if the 90% CI of the mean lies within ±margin (§15.8 equivalence)."""
    lo, hi = block_interval(values, 0.90, block, resamples, seed, np.mean, method)
    return bool(lo > -margin and hi < margin)


def sitting_statistic(data, method_scores, min_train, credit="proportional", K=100, rhos=RHOS):
    """Mean over test sittings and rho of AS for a scoring function; returns per-sitting means too."""
    per = []
    for s in range(min_train, len(data.sittings)):
        hist = History(data, s)
        scorer = get_scorer(data.sittings[s], data.topics, credit)
        curve = as_curve(scorer, method_scores(hist, np.random.default_rng(SEEDS["ties"] + s)), rhos, K=K)
        per.append(float(np.mean(list(curve.values()))))
    return float(np.mean(per)), per


def h1_statistic(data, min_train, credit="proportional", K=100, rhos=RHOS, random_draws=200):
    """Mean over test sittings and rho of AS_B1 - AS_random."""
    from lcbank.analysis.evaluate import as_random
    per = []
    for s in range(min_train, len(data.sittings)):
        hist = History(data, s)
        scorer = get_scorer(data.sittings[s], data.topics, credit)
        b1 = as_curve(scorer, METHODS["B1"](hist), rhos, K=K)
        rnd = as_random(scorer, rhos, draws=random_draws)
        per.append(float(np.mean([b1[r] - rnd[r] for r in rhos])))
    return float(np.mean(per)), per


def h2_statistic(data, min_train, credit="proportional", K=100, rhos=RHOS):
    """max over timing methods of mean(AS_m - AS_B1); returns (statistic, per-sitting values of the argmax method)."""
    base_mean, base_per = sitting_statistic(data, METHODS["B1"], min_train, credit, K, rhos)
    best, best_val, best_per = None, -np.inf, None
    for name in TIMING_METHODS:
        mean_m, per_m = sitting_statistic(data, METHODS[name], min_train, credit, K, rhos)
        if mean_m - base_mean > best_val:
            best, best_val = name, mean_m - base_mean
            best_per = [a - b for a, b in zip(per_m, base_per)]
    return float(best_val), best_per, best


def h3_beta(datasets, cap=5):
    """Coefficient on d (capped) in a logistic model of appears with dataset x topic intercepts (§15.8 H3)."""
    from sklearn.linear_model import LogisticRegression
    rows, y, keys = [], [], []
    for data in datasets:
        a = data.appears
        for k in range(a.shape[1]):
            first = np.flatnonzero(a[:, k])
            if len(first) == 0:
                continue
            for i in range(first[0] + 1, a.shape[0]):
                prev = np.flatnonzero(a[:i, k])
                d = min(i - prev[-1], cap)
                rows.append((f"{data.name}:{k}", d))
                y.append(int(a[i, k]))
    if not rows:
        return float("nan")
    keys = sorted({r[0] for r in rows})
    idx = {k: j for j, k in enumerate(keys)}
    X = np.zeros((len(rows), len(keys) + 1))
    for r, (key, d) in enumerate(rows):
        X[r, idx[key]] = 1.0
        X[r, -1] = d
    y = np.array(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(X, y)
    return float(model.coef_[0][-1])


def permute_within_sitting(data, rng):
    """Null N1: shuffle topic identities independently within each sitting (§15.8)."""
    appears = data.appears.copy()
    share = data.share.copy()
    for i in range(appears.shape[0]):
        perm = rng.permutation(appears.shape[1])
        appears[i] = appears[i][perm]
        share[i] = share[i][perm]
    return DatasetData(data.name, data.topics, data.sittings, appears=appears, share=share)


def permute_sitting_order(data, rng):
    """Null N2: shuffle the order of sittings, content and rules moving together (§15.8)."""
    order = rng.permutation(len(data.sittings))
    sittings = [data.sittings[i] for i in order]
    return DatasetData(data.name, data.topics, sittings, appears=data.appears[order], share=data.share[order])
