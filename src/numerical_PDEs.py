import numpy as np
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline

def american_put_fdm(S0, K, r, sigma, T, Nx=500, Nt=5000, x_range=3.0):
    """American put via explicit finite-difference PDE method in log-space.

    Solves the Black-Scholes PDE in log-coordinates (x = log S, tau = T - t):
        dV/dtau = (r - 0.5*sigma^2) dV/dx + (0.5*sigma^2) d2V/dx2 - r*V
        V(tau=0, x) = max(K - exp(x), 0)   (terminal payoff)
        V(tau, x) >= max(K - exp(x), 0)     (early-exercise constraint)

    The log-space transform makes the PDE constant-coefficient, simplifying
    the discretisation and CFL stability condition. Upwind differencing
    is used for the convection term (r - 0.5*sigma^2) dV/dx to avoid
    oscillations. The early-exercise constraint is applied as a max()
    floor after each explicit Euler time step.
    """
    x0 = np.log(S0)
    x_min, x_max = x0 - x_range, x0 + x_range
    x_grid = np.linspace(x_min, x_max, Nx + 1)
    dx = x_grid[1] - x_grid[0]
    dt = T / Nt
    S_grid = np.exp(x_grid)
    V = np.maximum(K - S_grid, 0.0)       # terminal payoff
    mu_x = r - 0.5 * sigma**2             # log-space drift
    cfl = dt * 0.5 * sigma**2 / dx**2     # CFL stability number

    # Time-stepping loop is inherently sequential (each step depends on
    # the previous one). The spatial discretisation is vectorised.
    for n in range(Nt):
        dV_dx = np.zeros_like(V)
        # Upwind: backward difference if mu_x > 0, forward if mu_x < 0
        if mu_x > 0:
            dV_dx[1:] = (V[1:] - V[:-1]) / dx
        else:
            dV_dx[:-1] = (V[1:] - V[:-1]) / dx
        d2V_dx2 = np.zeros_like(V)
        d2V_dx2[1:-1] = (V[2:] - 2*V[1:-1] + V[:-2]) / dx**2
        L_V = mu_x * dV_dx + 0.5 * sigma**2 * d2V_dx2 - r * V
        V_new = V + dt * L_V              # explicit Euler step
        V_new = np.maximum(V_new, K - S_grid)  # early-exercise constraint
        V = V_new

    price = np.interp(np.log(S0), x_grid, V)
    return price, cfl

def american_asian_put_fdm(S0, K, r, sigma, T, N_fix=50, Nx=200, Na=200):
    """American-Asian put via 2D FDM with three accuracy improvements
    over the explicit scheme:

    1. Crank-Nicolson time-stepping in S (theta=0.5): unconditionally
       stable, second-order in dt. No CFL restriction.
    2. Cubic spline interpolation in A at fixing dates: O(dA^4) accuracy
       per fixing vs O(dA^2) for linear interpolation.
    3. Tanh-stretched A-grid: concentrates grid points near the strike K
       where the early-exercise boundary lives.

    Between fixing dates: N_a independent 1D BS PDEs in log-S (A frozen).
    At fixing dates: A jumps via A_new = ((n-1)*A + S) / n and cubic-spline
    interpolation. Early-exercise constraint: V >= max(K - A, 0).
    """
    x0 = np.log(S0)
    x_grid = np.linspace(x0 - 2.5, x0 + 2.5, Nx + 1)
    dx = x_grid[1] - x_grid[0]
    S_grid = np.exp(x_grid)

    # Tanh-stretched A grid (concentrate near K)
    A_min, A_max = 0.01, 2.0 * K
    u = np.linspace(0, 1, Na + 1)
    stretch = 0.6
    A_grid = A_min + (A_max - A_min) * (
        1 + np.tanh(stretch * (2*u - 1)) / np.tanh(stretch)) / 2

    dt_fix = T / N_fix
    Nt_per_fix = 20   # CN is unconditionally stable, so fewer steps needed
    dt = dt_fix / Nt_per_fix
    mu_x = r - 0.5 * sigma**2
    sig2 = sigma**2

    # CN operator: (I - 0.5*dt*L) V_new = (I + 0.5*dt*L) V_old
    # Upwind for convection in log-S
    a_c = -mu_x/dx + 0.5*sig2/dx**2    # sub-diagonal
    b_c = mu_x/dx - sig2/dx**2 - r     # diagonal
    c_c = 0.5*sig2/dx**2              # super-diagonal

    ab = np.zeros((3, Nx + 1))
    ab[0, 1:]  = -0.5 * dt * c_c
    ab[1, :]   = 1.0 - 0.5 * dt * b_c
    ab[2, :-1] = -0.5 * dt * a_c
    ab[1, 0] = 1.0; ab[0, 1] = 0.0    # Dirichlet BC at S_min
    ab[1, -1] = 1.0; ab[2, -2] = 0.0  # Dirichlet BC at S_max

    # V[i, j] = V(S_i, A_j). At tau=0: max(K - A, 0)
    V = np.tile(np.maximum(K - A_grid, 0.0), (Nx + 1, 1))

    for n in range(N_fix, 0, -1):
        # Between fixings: CN in S for each A column
        for _ in range(Nt_per_fix):
            rhs = V.copy()
            rhs[1:-1, :] += 0.5 * dt * (a_c * V[:-2, :] + b_c * V[1:-1, :] + c_c * V[2:, :])
            rhs[0, :] = np.maximum(K - A_grid, 0)
            rhs[-1, :] = 0.0
            for j in range(Na + 1):
                V[:, j] = solve_banded((1, 1), ab, rhs[:, j])
            V = np.maximum(V, np.maximum(K - A_grid[np.newaxis, :], 0.0))

        # At fixing date: A jumps, cubic spline interpolation
        for i in range(Nx + 1):
            A_new = ((n - 1) * A_grid + S_grid[i]) / n
            A_new = np.clip(A_new, A_grid[0], A_grid[-1])
            cs = CubicSpline(A_grid, V[i, :])
            V[i, :] = np.nan_to_num(cs(A_new), nan=0.0)
        V = np.maximum(V, np.maximum(K - A_grid[np.newaxis, :], 0.0))

    j0 = np.argmin(np.abs(A_grid - S0))
    return np.interp(np.log(S0), x_grid, V[:, j0])


def asian_put_dp(S0, sigma, T, r, K=40.0, n_per_year=50, american=True,
                 nA=400, pts_per_std=2.5, x_std=5.0):
    n = int(round(T * n_per_year))
    dt = T / n
    disc = np.exp(-r * dt)
    s = sigma * np.sqrt(dt)
    m = (r - 0.5 * sigma**2) * dt

    # ---- log-price grid (uniform), P[i,k] = prob. of going from x_i to x_k in one step
    dx = s / pts_per_std
    half = x_std * sigma * np.sqrt(T) + abs(r - 0.5 * sigma**2) * T
    nx = int(np.ceil(2 * half / dx)) + 1
    x = np.log(S0) + (np.arange(nx) - nx // 2) * dx
    S = np.exp(x)
    d = x[None, :] - x[:, None] - m
    P = np.exp(-0.5 * (d / s) ** 2)
    P /= P.sum(axis=1, keepdims=True)
    i0 = nx // 2                                   # row of the initial price S0

    # ---- grid for the running average A (log-spaced)
    w = 4.0 * sigma * np.sqrt(T / 3.0)
    A = np.exp(np.linspace(np.log(S0) - w, np.log(S0) + w + r * T, nA))

    # ---- value at the last monitoring date, as a function of (S_n, A_{n-1}):
    #      A_n = ((n-1) A + S_n) / n
    def pre(U_next, j):
        """Value just BEFORE date j+1 is registered, as function of (S', A_j)."""
        out = np.empty_like(U_next)
        for k in range(nx):
            out[k] = np.interp((j * A + S[k]) / (j + 1), A, U_next[k])
        return out

    U = np.maximum(K - ((n - 1) * A[None, :] + S[:, None]) / n, 0.0)      # = Utilde_n
    for j in range(n - 1, 0, -1):                  # j = n-1, ..., 1
        C = disc * (P @ U)                         # continuation value at date j
        if american:
            C = np.maximum(C, np.maximum(K - A[None, :], 0.0))
        U = pre(C, j - 1)                          # Utilde_j(S', A_{j-1}); at j-1 = 0 the A-dependence drops
    # at this point U[k, :] = Utilde_1(S_k, .) and (j-1)*A + S) / j = S  ->  A-independent
    return disc * (P[i0] @ U[:, 0])


def asian_put_dp_err(S0, sigma, T, r, K=40.0, n_per_year=50, american=True,
                     nA=200, pts_per_std=2.5):
    """Price + numerical error estimate (discretisation error only, no sampling noise).

    * A-grid error: Richardson-type estimate from three runs with nA, 2nA, 4nA.
      The observed convergence order p = log2(d1/d2) is clipped to [1, 2]; if the
      differences are not monotone we fall back to p = 1 (most conservative).
    * log S-grid error: difference between pts_per_std and 1.6*pts_per_std at 2nA.
    Returns (price on the finest grid, estimated absolute error, Richardson-extrapolated price).
    """
    kw = dict(K=K, n_per_year=n_per_year, american=american)
    v0 = asian_put_dp(S0, sigma, T, r, nA=nA, pts_per_std=pts_per_std, **kw)
    v1 = asian_put_dp(S0, sigma, T, r, nA=2 * nA, pts_per_std=pts_per_std, **kw)
    v2 = asian_put_dp(S0, sigma, T, r, nA=4 * nA, pts_per_std=pts_per_std, **kw)
    d1, d2 = v1 - v0, v2 - v1
    p = np.clip(np.log2(d1 / d2), 1.0, 2.0) if (d1 * d2 > 0 and abs(d2) < abs(d1)) else 1.0
    err_A = abs(d2) / (2.0**p - 1.0)
    vx = asian_put_dp(S0, sigma, T, r, nA=2 * nA, pts_per_std=1.6 * pts_per_std, **kw)
    err_x = abs(vx - v1)
    return v2, err_A + err_x, v2 + d2 / (2.0**p - 1.0)

