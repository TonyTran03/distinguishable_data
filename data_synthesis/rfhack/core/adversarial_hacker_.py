import numpy as np
import pandas as pd
from collections import namedtuple
from .rf_wrapper import RFWrapper

AdversarialResult = namedtuple(
    "AdversarialResult",
    ["X_syn", "y_syn", "sep", "sep_min", "sep_max",
     "sigma", "iterations", "converged"],
)


class AdversarialHacker:
    """Binary-search sigma to drive a generator's RF separability to a target.

    gen_fn(seed) -> (X_syn, y_syn) returns synthetic samples
      the hacker adds Gaussian noise scaled by per-feature std of the real data.
    Separability = max(AUC, 1 - AUC) -> symmetric around 0.5.
    """

    def __init__(self, X_real, gen_fn, tol=0.01, max_iter=20, test_size=0.3, sigma_max=5.0, seed=42):
        self.X_real = np.asarray(X_real, dtype=np.float64)
        stds = self.X_real.std(axis=0)
        self.stds = np.where(stds == 0, 1.0, stds)
        self.gen_fn = gen_fn
        self.tol = tol
        self.max_iter = max_iter
        self.test_size = test_size
        self.sigma_max = sigma_max
        self.seed = seed
        self.feat_cols = [f"f{i}" for i in range(self.X_real.shape[1])]

    def _perturb(self, sigma, seed):
        X_syn, y_syn = self.gen_fn(seed)
        X_syn = np.asarray(X_syn, dtype=np.float64)
        if sigma > 0:
            rng = np.random.default_rng(seed + 500)
            X_syn = X_syn + rng.standard_normal(X_syn.shape) * self.stds * sigma
        return X_syn, y_syn

    def _separability(self, X_syn):
        real_df = pd.DataFrame(self.X_real, columns=self.feat_cols)
        real_df["target"] = 1
        syn_df = pd.DataFrame(X_syn, columns=self.feat_cols)
        syn_df["target"] = 0
        combined = pd.concat([real_df, syn_df], ignore_index=True)
        avg, mn, mx = RFWrapper.from_combined(combined, test_size=self.test_size)
        return max(avg, 1.0 - avg), mn, mx

    def hack(self, target_sep):
        lo, hi = 0.0, self.sigma_max
        best = None
        for i in range(self.max_iter):
            mid = (lo + hi) / 2
            X_syn, y_syn = self._perturb(mid, self.seed + i)
            sep, mn, mx = self._separability(X_syn)

            if best is None or abs(sep - target_sep) < abs(best.sep - target_sep):
                best = AdversarialResult(
                    X_syn, y_syn, sep, mn, mx, mid,
                    i + 1, abs(sep - target_sep) < self.tol,
                )
            if abs(sep - target_sep) < self.tol:
                break
            if sep < target_sep:
                lo = mid
            else:
                hi = mid
        return best
