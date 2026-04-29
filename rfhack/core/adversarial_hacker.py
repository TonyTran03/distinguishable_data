# adversarial_hacker.py is used for trying to reach a certain AUC.
import numpy as np
import pandas as pd
from collections import namedtuple
from .rf_wrapper import RFWrapper

AdversarialResult = namedtuple('result', ['df', 'auc', 'auc_min', 'auc_max', 'sigma', 'iterations'])


class AdversarialHacker:
    def __init__(self, df, n_samples=None, tol=0.005, max_iter=20, test_size=0.3):
        self.tol = tol
        self.max_iter = max_iter
        self.test_size = test_size
        self.real_df = df
        self.n_samples = n_samples or len(df)

    def _perturb(self, sigma):
        idx = np.random.choice(len(self.real_df), size=self.n_samples, replace=True)
        synth = self.real_df.iloc[idx].reset_index(drop=True).copy()
        noise = np.random.randn(*synth.shape) * synth.std().values * sigma
        synth += noise
        return synth

    def _combine(self, synth):
        real = self.real_df.copy()
        real['target'] = 1
        synth = synth.copy()
        synth['target'] = 0
        return pd.concat([real, synth], ignore_index=True)

    def hack(self, target_auc):
        lo, hi = 0.0, 5.0
        best = None
        for i in range(self.max_iter):
            mid = (lo + hi) / 2
            synth = self._perturb(mid)
            combined = self._combine(synth)
            avg, mn, mx = RFWrapper.from_combined(combined, test_size=self.test_size)
            if best is None or abs(avg - target_auc) < abs(best.auc - target_auc):
                best = AdversarialResult(synth, avg, mn, mx, mid, i + 1)
            if abs(avg - target_auc) < self.tol:
                break
            if avg < target_auc:
                lo = mid
            else:
                hi = mid
        return best