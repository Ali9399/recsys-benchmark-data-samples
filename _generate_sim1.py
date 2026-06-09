"""Generate the sim1 shareable benchmark CSVs.

Produces three artifacts in this directory:
  - training_sim1.csv   (200,000 rows)
  - test_sim1.csv       (20,000 rows)
  - readme_sim1.md      (DGP documentation; written separately)

DGP: 4-family concat-MLP with Poisson outcomes, target E[lambda] = 1
(sparse rate, ~63% zeros), canonical Zipf(1.0) per-family level
sampling, 2-hidden-layer truth network.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/alinasiriamini/coding/glm-regression/"
                "research/eb_embeddings")
from sim_Mfam_poisson_mlp import make_data


def main() -> None:
    out_dir = Path("/Users/alinasiriamini/coding/data-samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    # DGP parameters.
    Ks = (100, 500, 1000, 200)
    tau2 = (1.0, 3.0, 1.5, 0.3)
    d = 10
    H1, H2 = 32, 16
    target_lam = 1.0
    target_signal_var = 2.0
    zipf = 1.0
    seed = 42
    N_train = 200_000
    N_test = 20_000
    N_total = N_train + N_test

    print(f"  Generating sim1 (N={N_total}, M={len(Ks)}, "
          f"K={Ks}, lambda_target={target_lam}, Zipf={zipf}, "
          f"seed={seed})", flush=True)

    data = make_data(
        Ks, d, N_total, tau2,
        seed=seed, H1=H1, H2=H2,
        target_signal_var=target_signal_var,
        target_lam=target_lam, zipf=zipf,
        query_family=1,  # irrelevant for the CSV export
    )

    print(f"  empirical mean(y) = {data.y.mean():.4f}", flush=True)
    print(f"  empirical frac(y=0) = "
          f"{float((data.y == 0).mean()):.4f}", flush=True)
    print(f"  empirical mean(lambda_true) = "
          f"{data.lambda_true.mean():.4f}", flush=True)

    # First N_train rows -> training, last N_test rows -> test.
    # Both come from the SAME DGP, so the held-out rows test
    # generalisation to unseen INDEX COMBINATIONS (not new levels).
    def _write_csv(path: Path, rows_slice: slice) -> None:
        N_rows = (rows_slice.stop or N_total) - (rows_slice.start or 0)
        # Convert all-0-indexed family indices to 1-indexed.
        f1 = data.idxs[0][rows_slice] + 1
        f2 = data.idxs[1][rows_slice] + 1
        f3 = data.idxs[2][rows_slice] + 1
        f4 = data.idxs[3][rows_slice] + 1
        y = data.y[rows_slice].astype(np.int64)
        rate = data.lambda_true[rows_slice]
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["f1", "f2", "f3", "f4",
                             "observed", "true_rate"])
            for i in range(N_rows):
                writer.writerow([
                    int(f1[i]), int(f2[i]), int(f3[i]), int(f4[i]),
                    int(y[i]), f"{float(rate[i]):.6f}",
                ])
        print(f"    wrote {path} ({N_rows:,} rows)", flush=True)

    _write_csv(out_dir / "training_sim1.csv",
                slice(0, N_train))
    _write_csv(out_dir / "test_sim1.csv",
                slice(N_train, N_total))

    print("\n  Done.", flush=True)


if __name__ == "__main__":
    main()
