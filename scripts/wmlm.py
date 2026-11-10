# -*- coding: utf-8 -*-
"""
wmlm —— 两层随机截距（+随机斜率）模型的加权极大似然估计

模型： y_j = X_j β + Z_j u_j + e_j
       u_j ~ N(0, Ψ)                 Ψ = L L'（L 下三角）
       e_j ~ N(0, σ² diag(1/w_j))     w 为水平1 精度权重

为什么要自建：PISA 分析必须加权，而 statsmodels.MixedLM 不支持权重。
验证：w 全为 1 时应与 statsmodels.MixedLM(reml=False) 结果一致（见 14 脚本）。

参数化：k=1 → θ=[log σ², log ψ]；k=2 → θ=[log σ², log L11, L21, log L22]
"""
import numpy as np
from scipy.optimize import minimize


class TwoLevelML:
    def __init__(self, y, X, Z, gid, w=None, maxiter=500):
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.Z = np.asarray(Z, float)
        self.n, self.p = self.X.shape
        self.k = self.Z.shape[1]
        self.gid = np.asarray(gid)
        self.w = np.ones(self.n) if w is None else np.asarray(w, float)
        order = np.argsort(self.gid, kind='stable')
        gs = self.gid[order]
        uniq, starts = np.unique(gs, return_index=True)
        ends = np.r_[starts[1:], len(gs)]
        ys, Xs, Zs, ws = self.y[order], self.X[order], self.Z[order], self.w[order]
        self.J = len(uniq)
        self.groups = [(ys[a:b], Xs[a:b], Zs[a:b], ws[a:b]) for a, b in zip(starts, ends)]
        self.maxiter = maxiter
        self.theta0 = None

    # ---------- 参数 → L ----------
    def _L(self, theta):
        if self.k == 1:
            return np.array([[np.exp(theta[1])]])
        L = np.array([[np.exp(theta[1]), 0.0], [theta[2], np.exp(theta[3])]])
        return L

    def _default_theta(self):
        vy = np.var(self.y)
        if self.k == 1:
            return np.array([np.log(max(vy * .7, 1e-6)), np.log(max(vy * .3, 1e-6))])
        return np.array([np.log(max(vy * .7, 1e-6)), np.log(max(vy * .25, 1e-6)),
                         0.0, np.log(max(vy * .01, 1e-6))])

    # ---------- 给定方差参数，GLS 解 β 与对数似然 ----------
    def _gls(self, theta, want_xtvx=False):
        s2 = np.exp(theta[0])
        L = self._L(theta)
        Psi = L @ L.T
        Psi_inv = np.linalg.inv(Psi)
        logdet_Psi = 2 * np.log(np.abs(np.diag(L))).sum()
        XtVX = np.zeros((self.p, self.p))
        XtVy = np.zeros(self.p)
        yVy = 0.0
        logdet_all = 0.0
        for y, X, Z, w in self.groups:
            dinv = w / s2
            DZ = Z * dinv[:, None]
            A = Psi_inv + DZ.T @ Z
            Ai = np.linalg.inv(A)
            ViX = X * dinv[:, None] - DZ @ (Ai @ (DZ.T @ X))
            Vi_y = y * dinv - DZ @ (Ai @ (DZ.T @ y))
            XtVX += X.T @ ViX
            XtVy += X.T @ Vi_y
            yVy += y @ Vi_y
            logdet_all += np.log(s2 / w).sum() + logdet_Psi + np.linalg.slogdet(A)[1]
        beta = np.linalg.solve(XtVX, XtVy)
        # 剖面前似然：r'V^{-1}r = y'V'y - 2β'X'V'y + β'(X'V'X)β
        rVr = yVy - 2 * beta @ XtVy + beta @ (XtVX @ beta)
        ll = -0.5 * (logdet_all + rVr) - 0.5 * self.n * np.log(2 * np.pi)
        if want_xtvx:
            return beta, ll, XtVX
        return beta, ll

    def _negll(self, theta):
        try:
            _, ll = self._gls(theta)
        except (np.linalg.LinAlgError, ValueError):
            return 1e10
        return -ll if np.isfinite(ll) else 1e10

    def fit(self, theta0=None, warm=False):
        t0 = self._default_theta() if theta0 is None else np.asarray(theta0, float)
        if warm:
            res = minimize(self._negll, t0, method='BFGS', options=dict(maxiter=self.maxiter, gtol=1e-7))
        else:
            r1 = minimize(self._negll, t0, method='Nelder-Mead',
                          options=dict(maxiter=self.maxiter, xatol=1e-7, fatol=1e-7))
            res = minimize(self._negll, r1.x, method='BFGS',
                           options=dict(maxiter=self.maxiter, gtol=1e-7))
        self.theta = res.x
        self.ll = -res.fun
        beta, _, XtVX = self._gls(res.x, want_xtvx=True)
        self.beta = beta
        self.vcov_beta = np.linalg.inv(XtVX)
        L = self._L(res.x)
        Psi = L @ L.T
        self.var = dict(sigma2=float(np.exp(res.x[0])), Psi=Psi,
                        tau00=float(Psi[0, 0]), tau01=float(Psi[0, 1]) if self.k > 1 else np.nan,
                        tau11=float(Psi[1, 1]) if self.k > 1 else np.nan)
        return self

    def se(self):
        return np.sqrt(np.diag(self.vcov_beta))

    def lrt_slope(self, other):
        """与只有随机截距的模型比较，检验斜率方差（50:50 混合卡方）"""
        from scipy import stats
        lr = 2 * (self.ll - other.ll)
        p = 0.5 * stats.chi2.sf(max(lr, 0), 1) + 0.5 * (1.0 if lr <= 0 else 0.0)
        return lr, p


class TwoLevelMLFast(TwoLevelML):
    """向量化版本：把所有学校补齐成 (J, nmax, ·) 张量，单次 numpy 调用完成。
       与 TwoLevelML 数值等价（同一 θ 下对数似然相同）。"""

    def __init__(self, y, X, Z, gid, w=None, maxiter=500):
        super().__init__(y, X, Z, gid, w, maxiter)
        sizes = [len(g[0]) for g in self.groups]
        nmax = max(sizes)
        J = self.J
        self.nmax = nmax
        self.mask = np.zeros((J, nmax), bool)
        Yp = np.zeros((J, nmax)); Xp = np.zeros((J, nmax, self.p)); Zp = np.zeros((J, nmax, self.k))
        Wp = np.ones((J, nmax))
        for j, (yy, XX, ZZ, ww) in enumerate(self.groups):
            m = len(yy)
            self.mask[j, :m] = True
            Yp[j, :m] = yy; Xp[j, :m] = XX; Zp[j, :m] = ZZ; Wp[j, :m] = ww
        self.Yp, self.Xp, self.Zp, self.Wp = Yp, Xp, Zp, Wp

    def _gls(self, theta, want_xtvx=False):
        s2 = np.exp(theta[0])
        L = self._L(theta)
        Psi = L @ L.T
        Psi_inv = np.linalg.inv(Psi)
        logdet_Psi = 2 * np.log(np.abs(np.diag(L))).sum()
        dinv = self.Wp / s2
        WZ = self.Zp * dinv[..., None]
        A = Psi_inv[None] + np.einsum('jni,jnk->jik', WZ, self.Zp)
        Ai = np.linalg.inv(A)
        # D^{-1} Z A^{-1} Z' D^{-1} X
        t1 = np.einsum('jni,jnk->jik', WZ, self.Xp)          # (J,k,p)
        corrX = np.einsum('jni,jik->jnk', WZ, np.einsum('jik,jkl->jil', Ai, t1))
        ViX = self.Xp * dinv[..., None] - corrX
        t2 = np.einsum('jni,jn->ji', WZ, self.Yp)            # (J,k)
        corrY = np.einsum('jni,ji->jn', WZ, np.einsum('jik,jk->ji', Ai, t2))
        Vi_y = self.Yp * dinv - corrY
        XtVX = np.einsum('jni,jnk->ik', self.Xp, ViX)
        XtVy = np.einsum('jni,jn->i', self.Xp, Vi_y)
        yVy = float(np.einsum('jn,jn->', self.Yp, Vi_y))
        logd = np.where(self.mask, np.log(s2 / self.Wp), 0.0).sum()
        logdetA = np.linalg.slogdet(A)[1].sum()
        logdet_all = logd + logdet_Psi * self.J + logdetA
        beta = np.linalg.solve(XtVX, XtVy)
        rVr = yVy - 2 * beta @ XtVy + beta @ (XtVX @ beta)
        ll = -0.5 * (logdet_all + rVr) - 0.5 * self.n * np.log(2 * np.pi)
        if want_xtvx:
            return beta, ll, XtVX
        return beta, ll
