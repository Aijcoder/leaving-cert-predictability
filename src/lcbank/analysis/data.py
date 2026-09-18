"""Phase 8: analysis data structures with leakage-proof access (§15.2) and the R1 guard.

A DatasetData holds one dataset's sittings in year order: the rules tree and leaves of each sitting, the topic list
(codebook minus UNCLEAR) and the topic x sitting tables. History(data).before(s) is the only access allowed while
evaluating sitting s; touching sitting index >= s raises LeakageError.
"""
import json
from dataclasses import dataclass, field

import numpy as np

from lcbank.common import config
from lcbank.common.paths import DATA_DERIVED
from lcbank.structure.rules import all_compulsory, item_ids, max_score

APPEARS_THRESHOLD = 0.015  # §14; sensitivity 0.0075 and 0.03


class LeakageError(RuntimeError):
    """Raised when evaluation code tries to read a sitting at or after the test sitting."""


class ForecastError(RuntimeError):
    """Raised when a target sitting is after the last available sitting (R1)."""


@dataclass
class Sitting:
    year: int
    root: dict
    items: list           # dicts: item_id, marks, topics [{topic_id, weight}]
    max_score: float
    max_score_all: float  # same tree with every choice replaced by `all` (SQ1)

    @property
    def marks(self):
        return {i["item_id"]: i["marks"] for i in self.items}


@dataclass
class DatasetData:
    name: str
    topics: list                     # topic ids, UNCLEAR excluded
    sittings: list                   # Sitting, ordered by year
    appears: np.ndarray = field(default=None)   # (n_sittings, n_topics) bool
    share: np.ndarray = field(default=None)     # (n_sittings, n_topics) float

    def __post_init__(self):
        if self.appears is None:
            self.share = np.array([self._shares(s) for s in self.sittings])
            self.appears = self.share >= APPEARS_THRESHOLD

    def _shares(self, sitting):
        idx = {t: k for k, t in enumerate(self.topics)}
        out = np.zeros(len(self.topics))
        for it in sitting.items:
            for w in it["topics"] or []:
                k = idx.get(w["topic_id"])
                if k is not None:
                    out[k] += it["marks"] * w["weight"] / 100.0
        return out / sitting.max_score

    @property
    def years(self):
        return [s.year for s in self.sittings]

    def index_of(self, year):
        return self.years.index(year)

    def with_threshold(self, threshold):
        d = DatasetData(self.name, self.topics, self.sittings, appears=self.share >= threshold, share=self.share)
        return d


class History:
    """Read-only view of everything strictly before test sitting index `s` (§15.2)."""

    def __init__(self, data, s):
        if s > data.appears.shape[0]:
            raise LeakageError(f"sitting index {s} beyond the data")
        self._data = data
        self._s = s

    @property
    def n(self):
        return self._s

    @property
    def topics(self):
        return self._data.topics

    @property
    def appears(self):
        return self._data.appears[:self._s]

    @property
    def share(self):
        return self._data.share[:self._s]

    def sitting(self, i):
        if i >= self._s:
            raise LeakageError(f"sitting index {i} is not before the test sitting {self._s}")
        return self._data.sittings[i]

    def gaps(self):
        """d = sittings since last appearance, counted at the test sitting; 0 if never appeared."""
        a = self.appears
        out = np.zeros(a.shape[1], dtype=int)
        for k in range(a.shape[1]):
            idx = np.flatnonzero(a[:, k])
            out[k] = (a.shape[0] - idx[-1]) if len(idx) else 0
        return out


def check_target(year):
    """R1: no method may target a sitting after the last available sitting."""
    last = config.last_available_sitting()
    if year > last:
        raise ForecastError(f"target sitting {year} is after the last available sitting {last}: forecasting is not allowed")
    return year


def load_dataset(name, labelled=True, sitting_types=("main",), include_excluded=False):
    """Build a DatasetData from data_derived (labelled items + rules trees).

    include_excluded is for descriptive tables only (Phase 7). The analysis engine always uses the
    default, so a sitting marked analysis_excluded can never enter an evaluation or a forecast.
    """
    cb_name = {"PHY": "PHY", "PM": "PM", "PMT": "PM", "AM2": "AM2"}[name]
    cb = json.loads((DATA_DERIVED / "codebooks" / f"{cb_name}_v1.json").read_text())
    topics = [t["topic_id"] for t in cb["topics"]]
    src = DATA_DERIVED / "items" / (f"items_labelled_{name}.jsonl" if labelled else f"items_structured_{name}.jsonl")
    items = [json.loads(l) for l in open(src, encoding="utf-8")]
    by_year = {}
    for it in items:
        if it["sitting_type"] not in sitting_types or (it.get("analysis_excluded") and not include_excluded):
            continue
        by_year.setdefault(it["year"], []).append(it)
    sittings = []
    for year in sorted(by_year):
        roots = []
        for paper in sorted({it["paper"] for it in by_year[year]}):
            rid = f"{name}-{year}-P{paper}"
            roots.append(json.loads((DATA_DERIVED / "rules" / f"{rid}.json").read_text())["root"])
        root = roots[0] if len(roots) == 1 else {"type": "all", "children": roots,
                                                 "source_text": "Paper 1 and Paper 2 form one sitting"}
        its = by_year[year]
        marks = {i["item_id"]: i["marks"] for i in its}
        missing = set(item_ids(root)) - set(marks)
        if missing:
            raise ValueError(f"{name} {year}: {len(missing)} leaves in the rules tree have no item, e.g. {sorted(missing)[:3]}")
        sittings.append(Sitting(year=year, root=root, items=its, max_score=max_score(root, marks),
                                max_score_all=max_score(all_compulsory(root), marks)))
    return DatasetData(name=name, topics=topics, sittings=sittings)
