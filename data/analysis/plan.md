# Analysis plan — retrospective predictability of Leaving Certificate topics

Locked before any real-data result (WORKFLOW.md §15.1, R7). Everything below is fixed; any later change goes to
`DEVIATIONS.md` and makes the affected results exploratory.

**R1 restated:** this analysis is retrospective. No result, table or figure may rank, score or predict topics for a
sitting after 2026 (`last_available_sitting`). Every entry point asserts `target_sitting <= 2026`.

## 1. Datasets and roles

| Dataset | Sittings used | Role | Reason |
|---|---|---|---|
| PHY (Physics, 1999 syllabus) | 2002–2026 main sittings, 25 complete | **Formal (confirmatory)** | Gate passed: 25 ≥ 15 sittings with paper + scheme |
| PM (Mathematics, Project Maths) | 2014–2026 main sittings, 13 complete | Exploratory | §15: PM is always exploratory |
| PMT (Mathematics 2012–2013) | — | Not analysed | Bank only (§2) |
| AM2 (Applied Maths 2021 spec) | — | Not analysed | Bank only (§2) |
| AM1 (Applied Maths, old syllabus) | — | Dropped | Owner decision 2026-09-17 (DECISIONS D-017, DEVIATIONS EX-004) |

- **C = 1** (formal datasets). Holm correction family size = 2C + 1 = 3.
- Excluded from every analysis run: deferred sittings, the 2012–2013 "Project Maths variant" papers, and any paper
  with `analysis_excluded = true` in the items file (currently none).
- A sitting is one year. For PM a sitting is Paper 1 + Paper 2 scored under one `all` rules root.

## 2. Unit of analysis and inputs

- Leaf items with `marks`, `topics[{topic_id, weight}]` (weights from the marking scheme, Appendix B.1), and the
  paper's rules tree (`data_derived/rules/`).
- Topic set per dataset = the frozen codebook (`PHY_v1` 38 topics, `PM_v1` 25 topics), excluding `UNCLEAR`.
  T = 38 (PHY), 25 (PM).
- Labels are frozen (sha256 in STATUS.md) before any real-data evaluation. Results carry the label
  "provisional pending human QA" until the human QA in §13 passes.

## 3. Topic × sitting tables (§14)

- `offered_marks[s,k]` = Σ over leaves of `marks × weight_k / 100` over all offered parts (choices included).
- `share[s,k]` = `offered_marks[s,k]` ÷ the paper's max score from the rules tree.
- `appears[s,k]` = `share[s,k] ≥ 0.015`. Sensitivity thresholds: 0.0075 and 0.03.

## 4. Rolling origin

- Sittings are ordered by year. Training for test sitting s = all earlier sittings of the same dataset.
- First test sitting: the earliest sitting with ≥ 8 training sittings (formal) or ≥ 5 (exploratory).
  - **PHY:** first test sitting **2010** (training 2002–2009); 17 test sittings 2010–2026.
  - **PM:** first test sitting **2019** (training 2014–2018); 8 test sittings 2019–2026.
- `History(dataset).before(s)` is the only data access during evaluation; reading sitting ≥ s raises an error.
- Development sittings (PHY 2002–2004, PM 2014–2016) are never test sittings; the training-window rule already
  guarantees this and the code asserts it.

## 5. Methods (Appendix B.3)

n = number of training sittings, a = appearances, d = sittings since last appearance.

| Label | Method |
|---|---|
| B0 | Random ordering of topics |
| B1 | Base rate (a + 1) / (n + 2) |
| S1 | Recency-weighted appearance, half-life 5 sittings |
| S2 | "Absent last sitting first": B1 + 1 if absent in the last training sitting, else B1 |
| M2 | Gap hazard logistic regression: one-hot topic + 1[d = 2] + 1[d ≥ 3] + 1[not yet appeared], `LogisticRegression(C=1.0, penalty="l2")`, fitted on training rows with sitting index ≥ 2 |
| M3 | Lag-1 logistic regression: one-hot topic + appeared-in-last-sitting |

Ties in the ranking are broken at random, averaged over K seeded draws: **K = 1000** for reported point estimates,
**K = 100** inside permutation and simulation loops, with the same K used for the observed statistic in that test.

## 6. Metric (Appendix B.2)

- Revise the top `floor(ρT)` topics for ρ ∈ {0.1, 0.2, …, 0.9}.
- Credit per leaf, **primary**: `marks × Σ weights of revised topics / 100`; **sensitivity**: strict (full marks only
  if every topic of the leaf is revised). `UNCLEAR` weight never earns credit.
- `AS` = rules-tree score with that credit ÷ rules-tree max score.
- Random reference `AS_random(ρ)` = mean AS over **1000** random topic sets of size `floor(ρT)` (seeded).
- Reported: AS curve per method and ρ, AS(0.5), and ρ80 = smallest ρ with AS ≥ 0.80 by linear interpolation on the
  grid, or ">0.9".
- Secondary metrics for `appears`: Brier skill score vs B1, log loss, top-k precision lift.

## 7. Decomposition (§15.6, per dataset, per ρ, mean over test sittings)

- **SQ1 question choice** = `AS_random(ρ) − AS_random,all-compulsory(ρ)`, where the second term evaluates the same
  paper with every `best_k` / `best_k_constrained` replaced by `all`, normalised by its own max score.
- **SQ2 frequency gain** = `AS_B1(ρ) − AS_random(ρ)`.
- **SQ3 timing gain** = `AS_nested-best-timing(ρ) − AS_B1(ρ)`. At each test sitting the nested best of {S1, S2, M2, M3}
  is the method with the highest mean AS on an inner rolling origin inside that sitting's training window; with fewer
  than 3 inner test points, use B1.
- **SQ4** descriptive comparison across datasets with bootstrap rank intervals; overlapping intervals are not ordered.

## 8. Hypotheses and inference (§15.8)

Formal datasets only (PHY). Holm correction over the 3 tests below. All permutation p-values use
`p = (1 + #{null ≥ observed}) / (1 + #perm)` with **2000** permutations and fixed seeds.

- **H1** (per formal dataset, one-sided): mean over test sittings and ρ of `AS_B1 − AS_random`.
  Null **N1**: within-sitting topic permutation (topic identities shuffled independently within each sitting), full re-run.
- **H2** (per formal dataset, one-sided): `max over {S1, S2, M2, M3}` of the mean over test sittings and ρ of `AS_m − AS_B1`.
  Null **N2**: sitting-order shuffle (each sitting's content and rules move together), full re-run including the max.
- **H3** (pooled formal datasets, two-sided): coefficient β on d (sittings since last appearance, capped at 5, rows before
  a topic's first appearance excluded) in a logistic model of `appears` with dataset×topic intercepts. Null **N2**,
  comparing |β|.

Intervals: moving-block bootstrap over test sittings, block length 2, **2000** resamples, 95% percentile CI.
Equivalence: a gain whose 90% CI lies within ±5 percentage points is reported as "practically negligible".

## 9. Seeds

| Purpose | Seed |
|---|---|
| Random topic sets (AS_random) | 20260101 |
| Tie-breaking | 20260102 |
| Permutation tests | 20260103 |
| Bootstrap | 20260104 |
| Simulator | 20260105 |

## 10. Simulator and controls (§15.7, must pass before real data)

- Setup per dataset: real topic count, rules trees, leaf marks and leaf–topic structure, real number of sittings.
  Synthetic `appears` sequences are drawn and mapped to weights.
- **G0:** per-topic appearance rates from Beta(2, 3); no temporal effect.
- **G1:** add `+δ · 1[d ≥ 3]` to the log-odds, δ ∈ {0.5, 1.0, 1.5}.
- 500 runs per scenario; 199 permutations inside the simulator (99 if the runtime estimate from 5 runs exceeds 12 h,
  logged in DECISIONS).
- Pass criteria: G0 false-positive rate of H2 and H3 at α = 0.05 between 2% and 8%; G1 at δ = 1.5 detected ≥ 80%;
  bootstrap 95% CI coverage between 90% and 98%. The minimum detectable δ is reported.
- If a control fails: stop, debug, fix, rerun the simulator, log it; never proceed with a failing control.

## 11. Sensitivity analyses (all exploratory)

1. Strict credit instead of proportional.
2. Appearance thresholds 0.0075 and 0.03.
3. Excluding sittings 2020–2022.
4. Dropping `needs_human` parts.
5. Codebook collapsed to L1.

## 12. Outputs

`analysis/results/*.csv|json` (AS curves with CIs, decomposition, ρ80 table, Holm table, sensitivity tables,
simulator report), `analysis/figures/` (AS curves per dataset, stacked decomposition at ρ = 0.5, label-agreement plot),
and `reports/analysis_report.md`, which must state:
*"Retrospective analysis of past papers only. No prediction of any future examination."*

Only the Holm table for H1–H3 on formal datasets is confirmatory; everything else is exploratory. PM is always
exploratory. Results carry "provisional pending human QA" until the human QA passes.
