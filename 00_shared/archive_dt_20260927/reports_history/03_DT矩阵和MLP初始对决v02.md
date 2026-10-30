# MLP vs DT Duel Environment v0.2 — Search-Corrected

## What changed

1. Removed all 5D Max/Sum entrants. Current six DT families are Max-DDD, Max-DDDD, Sum-DDD, Sum-DDDD, Count-DDTT, Count-DDDT.
2. Final G + output threshold are no longer left entirely to random evolution in one-stage search. The 242 distinct binary predicates realizable by a single D plus threshold are enumerated exactly and the best head is selected for every branch candidate.
3. Search is staged:
   - topology-capacity upper bound with arbitrary unary branch functions;
   - exact single-D library (739 distinct analog functions) for simple solutions;
   - deeper residual search with shallow/identity seeds, multi-scale random initialization, coordinate descent, and restarts.
4. `best-found error` is separated from `certified topology optimum`.

## Correction to sum_threshold
The proposed `max(0,x-7)` on both branches followed by `sum>=2` is not exact; e.g. (0,9) becomes (0,2) and is a false positive.
However the underlying claim is correct: a single D per branch is sufficient. Use f(x)=15-x on both branches, saturating sum, then classify `sum<=14`. Therefore Sum-DDD and Sum-DDDD both have exact 0-error solutions.

## Structured tasks — DT one-stage

| architecture | sum_threshold | compare | xor_hi | interval_or |
|---|---:|---:|---:|---:|
| Max-DDD | 27* | 32* | 64* | 0 |
| Max-DDDD | 27* | 27* | 64* | 0 |
| Sum-DDD | **0** | **0** | **0** | **0** |
| Sum-DDDD | **0** | **0** | **0** | **0** |
| Count-DDTT | **40†** | **40†** | **0†** | **0†** |
| Count-DDDT | **40†** | **40†** | **0†** | **0†** |

`*` best found among the current exact-forward searches; exact fit itself is impossible even if Max branches are relaxed to arbitrary 0..15 unary functions for sum_threshold / compare / xor_hi. The listed nonzero minimum is not yet certified for Max.

`†` globally certified for the one-stage count topology after relaxing each branch to an arbitrary binary unary function. Thus 40 is a topology limit, not a training failure.

## Structured MLP ladder (best of restarts)

| architecture | sum_threshold | compare | xor_hi | interval_or |
|---|---:|---:|---:|---:|
| MLP h0 | 0 | 0 | 93 | 90 |
| MLP tanh h1 | 0 | 0 | 85 | 77 |
| MLP tanh h2 | 0 | 0 | 42 | 52 |
| MLP tanh h3 | 0 | 0 | 0 | 22 |
| MLP tanh h4 | 0 | 0 | 0 | 0 |
| MLP ReLU h3 | 0 | 0 | 0 | 53 |
| MLP ReLU h4 | 0 | 0 | 0 | 5 |

On these four hand-built tasks Sum-DDD is already too strong for the tasks to distinguish 3D from 4D.

## Random balanced truth-table pressure test
Eight independent 128/128 random binary truth tables, 256 states each. Best of 3 DT restarts; MLP h8 best of 4 restarts.

| architecture | mean errors | median | min | max |
|---|---:|---:|---:|---:|
| MLP tanh h8 | 85.875 | 86.0 | 76 | 93 |
| Sum-DDDD | 87.125 | 87.0 | 82 | 92 |
| Sum-DDD | 87.500 | 88.0 | 82 | 94 |
| Count-DDDT | 92.625 | 95.0 | 84 | 98 |
| Count-DDTT | 92.875 | 93.0 | 84 | 99 |
| Max-DDDD | 93.250 | 96.0 | 86 | 98 |
| Max-DDD | 93.750 | 96.0 | 85 | 100 |
| MLP ReLU h8 | 98.875 | 99.5 | 89 | 109 |

These are optimization results, not exact capacity rankings. The important 3D-vs-4D observation is that Sum-DDDD beats Sum-DDD by only 0.375 errors on average across these eight random tables, while individual tables sometimes favor 3D. The fourth D therefore has no clear practical advantage yet.

## Current interpretation
- Sum is the most general one-stage aggregation among the tested structured tasks and currently the strongest DT family in random-table fitting.
- Count has a very clean topology but a hard one-stage representational ceiling on relational/diagonal tasks.
- Max is appropriate for OR/region-like tasks but has strong one-stage structural limitations.
- 3D and 4D must remain separate contestants. 5D is removed.
- Future two-stage search should be initialized by embedding a solved one-stage network into `2->h->1`, so adding a stage can never appear worse merely because initialization failed.
