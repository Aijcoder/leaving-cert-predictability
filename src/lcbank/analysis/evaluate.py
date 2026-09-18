"""Phase 8: attemptable share AS(rho), rolling-origin evaluation and the SQ decomposition (§15.5-15.6, B.2).

AS(rho) = rules-tree score with topic credit / rules-tree max. Ranking ties are broken at random and averaged over
K seeded draws (§15.4): K = 1000 for reported estimates, 100 inside permutation and simulation loops.
"""
import numpy as np

from lcbank.analysis.data import History, check_target
from lcbank.analysis.methods import METHODS, TIMING_METHODS, b1_base_rate
from lcbank.structure.rules import all_compulsory, compile_tree

RHOS = tuple(round(0.1 * i, 1) for i in range(1, 10))
SEEDS = {"random_sets": 20260101, "ties": 20260102, "permutation": 20260103, "bootstrap": 20260104, "simulator": 20260105}


class SittingScorer:
    """Credit and AS for one sitting; leaf topic weights are cached as arrays for speed."""

    def __init__(self, sitting, topics, credit="proportional"):
        self.sitting = sitting
        self.credit = credit
        idx = {t: k for k, t in enumerate(topics)}
        self.item_ids = [i["item_id"] for i in sitting.items]
        self.marks = np.array([float(i["marks"]) for i in sitting.items])
        self.weights = np.zeros((len(sitting.items), len(topics)))
        self.has_unclear = np.zeros(len(sitting.items), dtype=bool)
        for r, it in enumerate(sitting.items):
            for w in it["topics"] or []:
                k = idx.get(w["topic_id"])
                if k is None:
                    self.has_unclear[r] = True
                else:
                    self.weights[r, k] += w["weight"] / 100.0
        self.root_all = all_compulsory(sitting.root)
        self._f = compile_tree(sitting.root, self.item_ids)
        self._f_all = compile_tree(self.root_all, self.item_ids)

    def credits_many(self, masks):
        """masks: (K, n_topics) of 0/1 -> credit matrix (K, n_items)."""
        masks = np.atleast_2d(np.asarray(masks, dtype=float))
        if self.credit == "proportional":
            return self.marks[None, :] * (masks @ self.weights.T)
        labelled = (self.weights.sum(axis=1) > 0) & ~self.has_unclear
        missed = (1.0 - masks) @ self.weights.T          # weight of unrevised topics per leaf
        covered = labelled[None, :] & (missed <= 1e-9)
        return self.marks[None, :] * covered

    def AS_many(self, masks, compulsory=False):
        cred = self.credits_many(masks)
        f = self._f_all if compulsory else self._f
        denom = self.sitting.max_score_all if compulsory else self.sitting.max_score
        return f(cred) / denom

    def AS(self, revised_mask, compulsory=False):
        return float(self.AS_many(np.atleast_2d(revised_mask), compulsory)[0])


_SCORER_CACHE = {}


def get_scorer(sitting, topics, credit="proportional"):
    """SittingScorer per (sitting, credit), cached: permutations reuse the same Sitting objects."""
    key = (id(sitting), credit, id(topics))
    hit = _SCORER_CACHE.get(key)
    if hit is None:
        hit = _SCORER_CACHE[key] = SittingScorer(sitting, topics, credit)
    return hit


def revised_masks(scores, n_topics, rhos, K, rng):
    """For each rho, K masks of the top floor(rho*T) topics, ties broken at random (§15.4).

    The K tie-break orders are drawn once and shared across rho: the ranking is the same, only the cut-off moves.
    """
    scores = np.asarray(scores, dtype=float)
    jitter = rng.random((K, n_topics))
    orders = np.lexsort((jitter, np.repeat(-scores[None, :], K, axis=0)), axis=-1)   # (K, T)
    rows = np.repeat(np.arange(K)[:, None], n_topics, axis=1)
    out = {}
    for rho in rhos:
        n = int(np.floor(rho * n_topics))
        masks = np.zeros((K, n_topics))
        if n:
            masks[rows[:, :n], orders[:, :n]] = 1.0
        out[rho] = masks
    return out


def as_curve(scorer, scores, rhos=RHOS, K=1000, seed=SEEDS["ties"], compulsory=False):
    rng = np.random.default_rng(seed)
    T = scorer.weights.shape[1]
    masks = revised_masks(np.asarray(scores, dtype=float), T, rhos, K, rng)
    return {rho: float(np.mean(scorer.AS_many(masks[rho], compulsory))) for rho in rhos}


def as_random(scorer, rhos=RHOS, draws=1000, seed=SEEDS["random_sets"], compulsory=False):
    """Mean AS over `draws` random topic sets of the same size (§15.5)."""
    rng = np.random.default_rng(seed)
    T = scorer.weights.shape[1]
    out = {}
    for rho in rhos:
        n = int(np.floor(rho * T))
        masks = np.zeros((draws, T))
        for d in range(draws):
            masks[d, rng.choice(T, size=n, replace=False)] = 1.0
        out[rho] = float(np.mean(scorer.AS_many(masks, compulsory)))
    return out


def first_test_index(data, min_train):
    return min_train


def nested_best_timing(data, s, credit, K, rhos):
    """Method with the highest mean AS on an inner rolling origin inside the training window (§15.6 SQ3)."""
    inner_start = max(2, s - 6)
    inner = [i for i in range(inner_start, s) if i >= 3]
    if len(inner) < 3:
        return "B1"
    best, best_val = "B1", -np.inf
    for name in TIMING_METHODS:
        vals = []
        for i in inner:
            hist = History(data, i)
            scorer = get_scorer(data.sittings[i], data.topics, credit)
            scores = METHODS[name](hist, np.random.default_rng(SEEDS["ties"] + i))
            curve = as_curve(scorer, scores, rhos, K=K)
            vals.append(np.mean(list(curve.values())))
        if np.mean(vals) > best_val:
            best, best_val = name, float(np.mean(vals))
    return best


def evaluate_dataset(data, min_train, credit="proportional", K=1000, rhos=RHOS, random_draws=1000,
                     methods=("B0", "B1", "S1", "S2", "M2", "M3"), with_random=True, with_nested=True):
    """Rolling-origin evaluation. Returns rows: sitting, method, rho, AS (plus AS_random and the all-compulsory
    random reference used by SQ1)."""
    rows = []
    for s in range(first_test_index(data, min_train), len(data.sittings)):
        sitting = data.sittings[s]
        check_target(sitting.year)
        hist = History(data, s)
        scorer = get_scorer(sitting, data.topics, credit)
        for name in methods:
            scores = METHODS[name](hist, np.random.default_rng(SEEDS["ties"] + s))
            for rho, val in as_curve(scorer, scores, rhos, K=K).items():
                rows.append({"dataset": data.name, "year": sitting.year, "method": name, "rho": rho, "AS": val})
        if with_random:
            for rho, val in as_random(scorer, rhos, draws=random_draws).items():
                rows.append({"dataset": data.name, "year": sitting.year, "method": "RANDOM", "rho": rho, "AS": val})
            for rho, val in as_random(scorer, rhos, draws=random_draws, compulsory=True).items():
                rows.append({"dataset": data.name, "year": sitting.year, "method": "RANDOM_ALL_COMPULSORY",
                             "rho": rho, "AS": val})
        if with_nested:
            name = nested_best_timing(data, s, credit, K=min(K, 100), rhos=rhos)
            scores = METHODS[name](hist, np.random.default_rng(SEEDS["ties"] + s))
            for rho, val in as_curve(scorer, scores, rhos, K=K).items():
                rows.append({"dataset": data.name, "year": sitting.year, "method": "NESTED_TIMING", "rho": rho,
                             "AS": val, "chosen": name})
    return rows


def mean_by(rows, method, field="AS"):
    vals = [r[field] for r in rows if r["method"] == method]
    return float(np.mean(vals)) if vals else float("nan")


def decomposition(rows):
    """SQ1 choice, SQ2 frequency, SQ3 timing (mean over test sittings and rho)."""
    return {"SQ1_question_choice": mean_by(rows, "RANDOM") - mean_by(rows, "RANDOM_ALL_COMPULSORY"),
            "SQ2_frequency_gain": mean_by(rows, "B1") - mean_by(rows, "RANDOM"),
            "SQ3_timing_gain": mean_by(rows, "NESTED_TIMING") - mean_by(rows, "B1")}


def rho80(curve):
    """Smallest rho reaching AS >= 0.80 by linear interpolation on the grid, else '>0.9'."""
    items = sorted(curve.items())
    for (r0, a0), (r1, a1) in zip(items, items[1:]):
        if a0 < 0.8 <= a1:
            return round(r0 + (0.8 - a0) * (r1 - r0) / (a1 - a0), 3)
    return items[0][0] if items[0][1] >= 0.8 else ">0.9"


def brier_skill(scores, truth, base):
    """Brier skill score of `scores` against `base` for the binary appears vector."""
    def brier(p):
        p = np.clip(np.asarray(p, dtype=float), 0, 1)
        return float(np.mean((p - truth) ** 2))
    b_base = brier(base)
    return float("nan") if b_base == 0 else 1 - brier(scores) / b_base
