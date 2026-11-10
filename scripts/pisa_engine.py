# -*- coding: utf-8 -*-
"""pisa_engine -- PISA design-based inference engine (plausible values + BRR-Fay).

Conventions:
  - Each of the 10 plausible values (PVs) is estimated separately and the results
    are combined with Rubin's rules.
  - 80 BRR-Fay replicate weights (Fay coefficient rho = 0.5):
    Var = 1/(R(1-rho)^2) * sum_r (theta_r - theta)^2
  - Total variance T = mean sampling variance + (1 + 1/M) * between-PV variance
  - Degrees of freedom follow Rubin (1987)
"""
import numpy as np
import pandas as pd
from scipy import stats

FAY = 0.5


def _wls(X, y, w):
    Xt = X.T * w
    return np.linalg.solve(Xt @ X + 1e-10 * np.eye(X.shape[1]), Xt @ y)


class PISAEngine:
    def __init__(self, df, pv_prefix='PV', outcome='SCIE', n_pv=10,
                 wt='W_FSTUWT', rep_prefix='W_FSTURWT', n_rep=80):
        self.df = df.reset_index(drop=True)
        self.pv = [f'{pv_prefix}{i}{outcome}' for i in range(1, n_pv + 1)]
        self.wt = df[wt].values
        self.reps = [df[f'{rep_prefix}{i}'].values for i in range(1, n_rep + 1)]
        self.n_pv, self.n_rep = n_pv, n_rep
        self.n = len(df)

    def fit(self, cols, name='', cluster=None):
        """Return a DataFrame with columns term/b/se/p/df."""
        X = np.column_stack([np.ones(self.n)] + [self.df[c].values for c in cols])
        names = ['(截距)'] + list(cols)
        k = X.shape[1]
        pts, samp = [], []
        for pv in self.pv:
            y = self.df[pv].values
            b0 = _wls(X, y, self.wt)
            reps = np.array([_wls(X, y, w) for w in self.reps])
            v = ((reps - b0) ** 2).sum(axis=0) / (self.n_rep * (1 - FAY) ** 2)
            pts.append(b0)
            samp.append(v)
        P = np.array(pts)
        qbar = P.mean(axis=0)
        Ubar = np.mean(samp, axis=0)
        B = P.var(axis=0, ddof=1)
        T = Ubar + (1 + 1 / self.n_pv) * B
        se = np.sqrt(np.maximum(T, 1e-12))
        dfree = np.where(B > 1e-12,
                         (self.n_pv - 1) * (1 + Ubar / ((1 + 1 / self.n_pv) * np.maximum(B, 1e-12))) ** 2,
                         1e6)
        t = qbar / se
        p = 2 * stats.t.sf(np.abs(t), df=dfree)
        out = pd.DataFrame(dict(model=name, term=names, b=qbar, se=se, p=p, df=dfree))
        out['t'] = t
        return out

    def r2(self, cols_base, cols_full):
        """Incremental R-squared (weighted, averaged over PVs)."""
        def r2_of(cols):
            X = np.column_stack([np.ones(self.n)] + [self.df[c].values for c in cols])
            vals = []
            for pv in self.pv:
                y = self.df[pv].values
                w = self.wt
                b = _wls(X, y, w)
                r = y - X @ b
                ybar = np.average(y, weights=w)
                ss_res = np.sum(w * r ** 2)
                ss_tot = np.sum(w * (y - ybar) ** 2)
                vals.append(1 - ss_res / ss_tot)
            return np.mean(vals)
        a = r2_of(cols_base)
        b = r2_of(cols_full)
        return a, b, b - a
