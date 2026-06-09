# Simulation 1 — Sparse-Poisson 4-Feature Concat-MLP

This directory holds a synthetic recommender-style benchmark generated
by a 4-feature concat-MLP with Poisson count outcomes. Anyone can
download the two CSV files, fit a model of their choice on the
training rows, and evaluate it on the held-out test rows.

## Files

| file | rows | description |
|---|---|---|
| `training_sim1.csv` | 200,000 | training set |
| `test_sim1.csv` | 20,000 | held-out test set |

Both files share the same column schema and originate from the same
data-generating process (DGP); the held-out rows differ only in the
random index combinations sampled.

## Column schema

| column | type | description |
|---|---|---|
| `f1` | integer in `[1, 100]` | level index of feature 1 |
| `f2` | integer in `[1, 500]` | level index of feature 2 |
| `f3` | integer in `[1, 1000]` | level index of feature 3 |
| `f4` | integer in `[1, 200]` | level index of feature 4 |
| `observed` | non-negative integer | observed Poisson count `y` |
| `true_rate` | positive real | latent rate `λ` such that `y ~ Poisson(λ)` |

`true_rate` is the **expected count** for that row under the DGP. It
is included so anyone evaluating a fitted model can compare predicted
rate to the truth on held-out rows without having to retrain on the
test set.

## Data-generating process

The DGP is a concat-MLP that produces a log-rate `f(x)` from the four
categorical features, then samples `y ~ Poisson(exp(f(x)))`. The
ground-truth function and parameters are as follows.

### Feature level counts and embeddings

| feature | `K_m` (levels) | `d_m` | true `τ²_m` (per-coord embedding variance) |
|---|---|---|---|
| f1 | 100 | 10 | 1.0 |
| f2 | 500 | 10 | 3.0 |
| f3 | 1000 | 10 | 1.5 |
| f4 | 200 | 10 | 0.3 |

Each level of each feature has its own learned 10-dimensional
embedding vector drawn from `N(0, τ²_m · I_d)`. The four `τ²` are
intentionally heterogeneous so a single global weight decay cannot
optimally regularise all four families.

### Architecture

```
z = concat(b_{f1}[f1], b_{f2}[f2], b_{f3}[f3], b_{f4}[f4])  ∈ R^40
h1 = tanh(W1 · z  + c1)                                       (W1: 32×40)
h2 = tanh(W2 · h1 + c2)                                       (W2: 16×32)
f  = w_out · h2 + d_out                                       (w_out: 16,)
λ  = exp(f)
y ~ Poisson(λ)
```

- `H1 = 32`, `H2 = 16`, `d = 10`.
- Output `f` is rescaled (via `w_out`) so `Var[f] = 2.0` exactly per seed.
- `d_out` includes an offset chosen so `E[λ] = 1.0` exactly.
- `tanh` activations everywhere.

### Per-row sampling

Each row's four feature levels are drawn independently from per-feature
Zipf distributions:

```
P(f_m = k)  ∝  1 / k^α    for k = 1, …, K_m,   α = 1.0
```

This is the canonical Zipf law — head levels appear many times,
tail levels appear once or twice in `N = 220,000` rows. It is a
synthetic analog of the cold-start regime in real recommenders.

### Outcome statistics

After auto-tuning the output offset, the empirical statistics
across the full 220,000 rows are:

| statistic | value |
|---|---|
| `mean(y)` | 0.9995 |
| `mean(λ)` | 1.0000 |
| `frac(y = 0)` | 0.6217 (typical for Poisson(1)) |
| `var(f)` | 2.000 |
| seed | 42 |

The DGP was generated with a single seed (42) so the dataset is
reproducible.

## Suggested evaluation protocol

For a regression / ranking benchmark on this data:

1. **Train** on `training_sim1.csv` using `(f1, f2, f3, f4, observed)`
   as input; ignore `true_rate` at train time.
2. **Predict** a rate `λ̂_i = exp(f̂_i)` for each row of
   `test_sim1.csv` using `(f1, f2, f3, f4)`.
3. **Evaluate** against the true held-out rate:
   - **Poisson NLL**: `mean( λ̂_i − observed_i · log(λ̂_i) )`
     (the standard Poisson loss on the test set).
   - **True-rate RMSE**: `sqrt( mean( (log(λ̂_i) − log(true_rate_i))² ) )`
     (compares predicted log-rate to ground-truth log-rate; works
     even when many `observed` are zero).
   - **Slate ranking reward**: see the parent paper's evaluation
     protocol; not directly computable from a CSV without
     re-constructing slate queries.

The 62% zero-fraction in `observed` means raw count-MSE is dominated
by zeros. The log-rate RMSE is a better quality signal on this DGP.

## Cold-start / held-out levels

In this version of the benchmark, every feature level appears in both
training and test rows (because `N = 220,000 >> K_max = 1000`). This
is a *warm-start* benchmark: the model has seen every level at least
once at training time.

For a cold-start variant (some levels held out entirely from training),
a follow-up `sim2` will be generated separately.

## How to load

Python with `pandas`:

```python
import pandas as pd
train = pd.read_csv("training_sim1.csv")
test  = pd.read_csv("test_sim1.csv")
```

R with `data.table`:

```r
library(data.table)
train <- fread("training_sim1.csv")
test  <- fread("test_sim1.csv")
```

The integer feature columns are 1-indexed (not 0-indexed). If your
implementation uses 0-indexed levels, subtract 1 from each `f*` column.

## License and provenance

These files were generated from a deterministic synthetic DGP and
contain no real user data. Anyone is free to use them for benchmark
comparisons, blog posts, papers, or teaching.

## Companion paper

The DGP family and the hierarchical-likelihood (HL) baseline used in
the companion paper "Self-Regularization of Embedding Layers in Deep
Neural Networks via Hierarchical Likelihood" are documented in the
research repository (private; will be linked when public).
