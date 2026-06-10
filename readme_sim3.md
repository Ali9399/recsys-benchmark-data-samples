# sim3 — sparse-Poisson 8-feature concat-MLP benchmark

A second benchmark in the same family as `sim1`, but **larger and
sparser** to push the gap between hierarchical-likelihood (HL) shrinkage
and tuned fixed-weight-decay (MLE+L2) further.

## Files

| file                  | rows    | size  |
|-----------------------|---------|-------|
| `training_sim3.csv`   | 400,000 | ~12 MB|
| `test_sim3.csv`       | 40,000  | ~1.2 MB|

Each CSV has 10 columns:

```
f1, f2, f3, f4, f5, f6, f7, f8, observed, true_rate
```

- `f1` through `f8` are **1-indexed integer level IDs** for eight
  categorical features. Their cardinalities are
  `(100, 500, 1000, 200, 100, 500, 1000, 200)`.
- `observed` is the integer Poisson count for the row.
- `true_rate` is the true Poisson rate $\lambda_n = \exp(f_n^\text{true})$
  used to draw `observed`. Shipped to 6 decimal places to enable
  RMSE-on-log-rate evaluation against the truth.

## Data-generating process

For each row $n = 1, \dots, 440{,}000$:

1. Sample feature levels $(i_{n,1}, \dots, i_{n,8})$ independently from
   a **Zipf$(\alpha = 1.0)$** distribution over the levels of each
   feature. Head levels appear orders of magnitude more often than
   tail levels — a synthetic analog of cold-start in recommender data.
2. Look up the row's eight 10-dim embeddings,
   $b^{(m)}_{i_{n,m}} \in \mathbb{R}^{10}$, drawn at simulation start
   from $\mathcal{N}(0, \tau_m^2 I_{10})$ with **per-feature scales**:

   | $m$       | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   |
   |-----------|-----|-----|-----|-----|-----|-----|-----|-----|
   | $K_m$     | 100 | 500 | 1000| 200 | 100 | 500 | 1000| 200 |
   | $\tau_m^2$ | 1.0 | 3.0 | 1.5 | 0.3 | 0.3 | 1.5 | 3.0 | 1.0 |

   This **swapped-pair design** ensures (i) eight independent scales,
   (ii) the same set $\{0.3, 1.0, 1.5, 3.0\}$ shows up twice with
   different cardinalities, so a single global weight decay must
   compromise across two different (K, $\tau^2$) regimes for each value.

3. Concatenate the eight embeddings into a single vector
   $z_n \in \mathbb{R}^{80}$.
4. Pass it through a **2-hidden-layer concat-MLP** with `tanh`
   activations:

   $$
   f_n^\text{true} \;=\; w_\text{out}^\top \tanh(W_2 \tanh(W_1 z_n + c_1) + c_2) + d_\text{out}
   $$

   where $W_1 \in \mathbb{R}^{32 \times 80}$, $W_2 \in \mathbb{R}^{16 \times 32}$,
   $w_\text{out} \in \mathbb{R}^{16}$. The biases $d_\text{out}$ and the
   per-layer scale of $w_\text{out}$ are auto-tuned at simulation start
   so that the empirical signal variance is $\mathrm{Var}(f^\text{true}) \approx 2$
   and the mean Poisson rate is $\mathbb{E}[\lambda] \approx 0.25$.
5. Draw $y_n \sim \mathrm{Poisson}(\lambda_n)$ with
   $\lambda_n = \exp(f_n^\text{true})$.

### Outcome statistics

| statistic                | value   |
|--------------------------|---------|
| mean(`observed`)         | 0.2498  |
| frac(`observed` == 0)    | 0.8318  |
| mean(`true_rate`)        | 0.2500  |
| seed                     | 42      |

About **83% of rows are zero**, vs ~62% in sim1. This regime is what
makes the parameter-recovery question hard: most rows carry no
information about the embedding magnitudes, so the learner must
extrapolate sharply.

## Reference benchmark (1 seed)

We fit four estimators on the 400,000 training rows and evaluated on
the 40,000 test rows using the same recipe across all arms (Adam, lr =
0.02, 2,000 steps, batch = 20,000 = 5% of train, drop+rescale scheme b
for the per-family upweighting, 200-step MLE warm-up at WD = 1.0
before turning on HL). Numbers below are **deterministic** given the
recipe — the only variance source is the model-init RNG (single seed
for this reference).

| Arm                            | Test Poisson NLL | RMSE vs true f |
|--------------------------------|------------------|----------------|
| MLE + fixed weight decay (1.0) | 0.4701           | 1.0284         |
| HL — shared $\tau$             | 0.4005           | 0.5402         |
| HL — per-family $\tau$         | 0.4008           | 0.5405         |
| MAP per-family (no APL term)   | 0.4284           | 0.7572         |

### Gap vs MLE + L2

| Metric      | sim1 gap (HL vs MLE) | sim3 gap (HL vs MLE) |
|-------------|---------------------:|---------------------:|
| Test NLL    | $-36.3\%$            | $-14.7\%$            |
| RMSE_f      | $-23.4\%$            | $\mathbf{-47.4\%}$   |
| RMSE_f (abs)| 0.137                | $\mathbf{0.488}$     |

Sim3 has **3.6× larger absolute RMSE_f gap** than sim1. The relative
NLL gap is smaller only because at $\lambda = 0.25$ the irreducible
Poisson-noise floor swamps the test NLL — there is much less variance
left for any estimator to capture. The RMSE_f metric isolates the
log-rate recovery and exposes the structural failure of a single
shared weight decay across eight asymmetric feature scales.

### Recovered per-family $\hat\tau$ (HL-per-family)

| family $m$                  | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   |
|-----------------------------|-----|-----|-----|-----|-----|-----|-----|-----|
| $\tau^2_\text{set}$         | 1.0 | 3.0 | 1.5 | 0.3 | 0.3 | 1.5 | 3.0 | 1.0 |
| $\hat\tau$ (HL-per-family)  | 0.31| 0.76| 0.67| 0.27| 0.24| 0.45| 0.77| 0.43|

Families 2 and 7 (true $\tau^2 = 3.0$) correctly receive the largest
$\hat\tau$; families 4 and 5 (true $\tau^2 = 0.3$) the smallest.
Absolute magnitudes are gauge-shifted by the multiplicative b–W₁
freedom in the concat-MLP, but ratios are identifiable.

`MAP per-family (no APL)` $\hat\tau_m$ collapses to ~0 in all
families — the well-known boundary pathology of MAP without the
profile-likelihood normalizer correction.

## Suggested evaluation protocol

For a regression / ranking benchmark on this data:

1. Train on `training_sim3.csv` only.
2. Compute on `test_sim3.csv`:
   - **Test Poisson NLL**:
     $\frac{1}{N_\text{test}} \sum_n \left(\hat\lambda_n - y_n \log \hat\lambda_n\right)$
     (drop the $\log y_n!$ constant; matches the table above).
   - **RMSE on log-rate vs truth**:
     $\sqrt{\frac{1}{N_\text{test}} \sum_n (\log \hat\lambda_n - \log \lambda_n^\text{true})^2}$.
3. Optionally evaluate **ranking** by holding `f1` fixed within a
   query and ranking candidates by $\hat\lambda$ — same protocol as
   sim1 (see `readme_sim1.md`).

## Loading

```python
import pandas as pd
train = pd.read_csv("training_sim3.csv")
test  = pd.read_csv("test_sim3.csv")
```

```r
library(data.table)
train <- fread("training_sim3.csv")
test  <- fread("test_sim3.csv")
```

The integer feature columns are 1-indexed (not 0-indexed). If your
implementation uses 0-indexed levels, subtract 1 from each `f*` column.

## License and provenance

These files were generated from a deterministic synthetic DGP and
contain no real user data. Anyone is free to use them for benchmark
comparisons, blog posts, papers, or teaching.
