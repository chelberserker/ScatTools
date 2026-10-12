import numpy as np
from functools import lru_cache
from scipy.fft import dst
from scipy.optimize import brentq, newton

_LAST_ALPHA = {'a': None}   # warm start for the self-consistency search across calls


@lru_cache(maxsize=8)
def _grid(n, dr):
    """Grid in units of the core diameter sigma, plus FFT prefactors."""
    r = (np.arange(n) + 1) * dr
    k = np.pi * (np.arange(n) + 1) / ((n + 1) * dr)
    dk = np.pi / ((n + 1) * dr)
    return r, k, 2 * np.pi * dr / k, dk / (4 * np.pi**2 * r)


def _anderson(g, x0, tol, maxit, beta=0.7, depth=6):
    """Fixed-point solver x = g(x) with Anderson (Type-II) acceleration."""
    x = x0.copy()
    gx = g(x)
    F, X = [], []
    for it in range(maxit):
        f = gx - x
        err = np.max(np.abs(f))
        if err < tol:
            return gx, it
        F.append(f); X.append(x.copy())
        if len(F) > depth + 1:
            F.pop(0); X.pop(0)
        if len(F) > 1:
            dF = np.array([F[i + 1] - F[i] for i in range(len(F) - 1)]).T
            dX = np.array([X[i + 1] - X[i] for i in range(len(X) - 1)]).T
            gam, *_ = np.linalg.lstsq(dF, f, rcond=None)
            xn = x + beta * f - (dX + beta * dF) @ gam
        else:
            xn = x + beta * f
        x = xn
        gx = g(x)
    return gx, maxit


def hard_core_soft_shell(q, R, nu, dR, eps=1.0, alpha=None,
                        n=4095, dr=0.02, tol=1e-10, maxit=400
                        ):
    """S(q) for a hard core (radius R) with a repulsive square shoulder
    (height eps in kT, out to centre distance 2(R+dR)), Rogers-Young closure.

        u(r)/kT = inf  (r < 2R) ;  eps  (2R <= r < 2(R+dR)) ;  0 (beyond)

    q     : scattering vector(s), inverse of the unit of R
    R     : hard-core radius
    nu    : volume fraction of the HARD CORES (n * 4/3 pi R^3)
    dR    : shell thickness (same unit as R)
    eps   : shoulder height in kT (>= 0)
    alpha : RY mixing parameter in units of 1/(2R). None -> chosen so that the
            virial and compressibility routes agree. 0 = PY, large = HNC.
    n, dr : grid size and spacing in units of 2R. n should be 2**k - 1 for fast
            FFTs. Lowest resolved q ~ pi/(n*dr*2R); lower q is held constant.

            for higher precision use n=8191, dr=0.01. This is about 3 times more accurate, but twice as slow.

    full_output : also return (alpha, residual of the consistency condition)
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    sig = 2.0 * R
    lam = (R + dR) / R
    rhos = 6.0 * nu / np.pi                      # rho * sigma^3
    r, k, Af, Ab = _grid(n, dr)

    # exp(-u/kT), averaged over each grid cell (steps at 1 and lam need not be on-grid)
    lo, hi = r - dr / 2, r + dr / 2
    w_core = np.clip((np.minimum(hi, 1.0) - lo) / dr, 0, 1)
    w_shell = np.clip((np.minimum(hi, lam) - np.maximum(lo, 1.0)) / dr, 0, 1)
    eb = np.exp(-eps)
    e = w_shell * eb + (1.0 - w_core - w_shell)
    r2dr = r**2 * dr

    def fwd(c):
        return Af * dst(r * c, type=1)

    def bwd(ch):
        return Ab * dst(k * ch, type=1)

    def solve(rs, a, g0):
        f = np.clip(-np.expm1(-a * r), 1e-12, None)
        def y_of(gam):
            return 1.0 + np.expm1(np.minimum(f * gam, 50.0)) / f
        def step(gam):
            ch = fwd(e * y_of(gam) - 1.0 - gam)
            return bwd(rs * ch**2 / (1.0 - rs * ch))
        gam, nit = _anderson(step, g0, tol, maxit)
        return gam, y_of(gam), nit

    def z_virial(nu_, y):                       # beta*P/rho (virial route)
        y1 = np.interp(1.0, r, y)
        yl = np.interp(lam, r, y)
        return 1.0 + 4.0 * nu_ * (eb * y1 + lam**3 * (1.0 - eb) * yl)

    state = {"gam": np.zeros(n)}

    def consistency(a):
        a = max(a, 1e-9)
        gam, y, _ = solve(rhos, a, state["gam"])
        state["gam"] = gam
        c = e * y - 1.0 - gam
        inv_s0 = 1.0 - 24.0 * nu * np.sum(r2dr * c)          # compressibility route
        d = 2e-3
        zs = []
        for s in (1 + d, 1 - d):
            g2, y2, _ = solve(rhos * s, a, gam)
            zs.append(s * nu * z_virial(s * nu, y2))
        dP = (zs[0] - zs[1]) / (2 * d * nu)                  # d(beta P)/d(rho)
        return dP - inv_s0

    if alpha is None:
        if eps == 0 and dR >= 0 and nu < 1e-4:
            alpha_ = 1.0                                       # near-ideal: closure irrelevant
        else:
            lo_a, hi_a = 1e-3, 30.0
            alpha_ = None
            if _LAST_ALPHA['a'] is not None:                   # fast path: secant from last alpha
                try:
                    a0 = _LAST_ALPHA['a']
                    a_ = newton(consistency, a0, x1=a0 * 1.15, tol=1e-3, maxiter=8)
                    if lo_a < a_ < hi_a:
                        alpha_ = a_
                except (RuntimeError, FloatingPointError):
                    pass
            if alpha_ is None:                                 # robust path: bracketing
                flo, fhi = consistency(lo_a), consistency(hi_a)
                if flo * fhi < 0:
                    alpha_ = brentq(consistency, lo_a, hi_a, xtol=1e-3, rtol=1e-3)
                else:                                          # no root: take the closer end
                    alpha_ = lo_a if abs(flo) < abs(fhi) else hi_a
            _LAST_ALPHA['a'] = alpha_
    else:
        alpha_ = float(alpha)

    gam, y, nit = solve(rhos, alpha_ if alpha_ > 0 else 1e-9, state["gam"])
    ch = fwd(e * y - 1.0 - gam)
    S = 1.0 / (1.0 - rhos * ch)
    Sq = np.interp(q * sig, k, S, left=S[0])


    return Sq