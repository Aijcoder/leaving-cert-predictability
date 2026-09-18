"""Phase 8: topic-ranking methods B0, B1, S1, S2, M2, M3 (Appendix B.3).

Each method takes a History (everything strictly before the test sitting) and returns one score per topic; the
ranking is descending, with ties broken at random (see evaluate.py). No method sees the test sitting.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

HALF_LIFE = 5.0


def b0_random(history, rng):
    return rng.random(len(history.topics))


def b1_base_rate(history, rng=None):
    a = history.appears.sum(axis=0)
    n = history.appears.shape[0]
    return (a + 1) / (n + 2)


def s1_recency(history, rng=None):
    a = history.appears
    n = a.shape[0]
    j = np.arange(n)
    w = 0.5 ** ((n - 1 - j) / HALF_LIFE)   # j = n-1 is the most recent training sitting
    return (a * w[:, None]).sum(axis=0) / w.sum()


def s2_absent_last(history, rng=None):
    base = b1_base_rate(history)
    last = history.appears[-1]
    return base + (~last).astype(float)


def _gap_matrix(appears):
    """d[i, k] = sittings since the last appearance of topic k before sitting i (0 = never appeared yet)."""
    n, T = appears.shape
    d = np.zeros((n, T), dtype=int)
    last = np.full(T, -1)
    for i in range(n):
        d[i] = np.where(last >= 0, i - last, 0)
        last = np.where(appears[i], i, last)
    return d


def _design_gap(appears, upto):
    """Rows (sitting i >= 2, topic k): one-hot topic, 1[d=2], 1[d>=3], 1[never appeared yet]."""
    a = appears[:upto]
    n, T = a.shape
    if n <= 2:
        return np.zeros((0, T + 3)), np.zeros(0, dtype=int)
    d = _gap_matrix(a)[2:]
    rows = n - 2
    X = np.zeros((rows * T, T + 3))
    X[np.arange(rows * T), np.tile(np.arange(T), rows)] = 1.0
    X[:, T] = (d == 2).reshape(-1)
    X[:, T + 1] = (d >= 3).reshape(-1)
    X[:, T + 2] = (d == 0).reshape(-1)
    return X, a[2:].reshape(-1).astype(int)


def _design_lag1(appears, upto):
    a = appears[:upto]
    n, T = a.shape
    if n <= 1:
        return np.zeros((0, T + 1)), np.zeros(0, dtype=int)
    rows = n - 1
    X = np.zeros((rows * T, T + 1))
    X[np.arange(rows * T), np.tile(np.arange(T), rows)] = 1.0
    X[:, T] = a[:-1].reshape(-1).astype(float)
    return X, a[1:].reshape(-1).astype(int)


def _fit_predict(X, y, X_test):
    if len(np.unique(y)) < 2:
        return np.full(len(X_test), float(y[0]) if len(y) else 0.5)
    # Appendix B.3 asks for LogisticRegression(C=1.0, penalty="l2"); scikit-learn 1.8 deprecated the `penalty`
    # argument and applies L2 by default, so C=1.0 with the default penalty is the same model.
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X, y)
    return model.predict_proba(X_test)[:, 1]


def m2_gap_hazard(history, rng=None):
    a = history.appears
    n, T = a.shape
    X, y = _design_gap(a, n)
    if len(X) == 0:
        return b1_base_rate(history)
    gaps = history.gaps()
    rows = []
    for k in range(T):
        onehot = np.zeros(T)
        onehot[k] = 1
        d = gaps[k]
        rows.append(np.concatenate([onehot, [float(d == 2), float(d >= 3), float(d == 0)]]))
    return _fit_predict(X, y, np.array(rows))


def m3_lag1(history, rng=None):
    a = history.appears
    n, T = a.shape
    X, y = _design_lag1(a, n)
    if len(X) == 0:
        return b1_base_rate(history)
    last = a[-1]
    rows = []
    for k in range(T):
        onehot = np.zeros(T)
        onehot[k] = 1
        rows.append(np.concatenate([onehot, [float(last[k])]]))
    return _fit_predict(X, y, np.array(rows))


METHODS = {"B0": b0_random, "B1": b1_base_rate, "S1": s1_recency, "S2": s2_absent_last,
           "M2": m2_gap_hazard, "M3": m3_lag1}
TIMING_METHODS = ["S1", "S2", "M2", "M3"]
