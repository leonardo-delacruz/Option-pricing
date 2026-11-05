
import numpy as np
from scipy.stats import norm

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

def american_asian_put_fdm(S0, K, r, sigma, T, N_fix=50, Nx=150, Na=120):
    """American-Asian put via 2D explicit FDM.

    State: (S, A) where A is the running arithmetic average.
    Between fixing dates: 1D BS PDE in log(S) for each frozen A
    (A does not change between discrete monitoring dates).
    At fixing dates: A jumps to ((n-1)*A_old + S) / n via interpolation.
    Early-exercise constraint: V >= max(K - A, 0) applied at each step.
    """
    x0 = np.log(S0)
    x_grid = np.linspace(x0 - 2.5, x0 + 2.5, Nx + 1)
    dx = x_grid[1] - x_grid[0]
    S_grid = np.exp(x_grid)
    A_grid = np.linspace(0.01, 2.0 * K, Na + 1)
    dt_fix = T / N_fix
    Nt_per_fix = max(int(np.ceil(0.4 / (0.5 * sigma**2 * dt_fix / dx**2))), 5)
    dt = dt_fix / Nt_per_fix
    mu_x = r - 0.5 * sigma**2

    # V[i, j] = V(S_i, A_j). At tau=0 (t=T): max(K - A, 0) for all S
    V = np.tile(np.maximum(K - A_grid, 0.0), (Nx + 1, 1))

    for n in range(N_fix, 0, -1):
        # Between fixings: 1D BS PDE in S (log-space), A frozen
        for _ in range(Nt_per_fix):
            dV_dx = np.zeros_like(V)
            dV_dx[1:, :] = (V[1:, :] - V[:-1, :]) / dx   # upwind
            d2V_dx2 = np.zeros_like(V)
            d2V_dx2[1:-1, :] = (V[2:, :] - 2*V[1:-1, :] + V[:-2, :]) / dx**2
            L_V = mu_x * dV_dx + 0.5 * sigma**2 * d2V_dx2 - r * V
            V = V + dt * L_V
            V[0, :] = np.maximum(K - A_grid, 0)   # deep ITM
            V[-1, :] = 0.0                        # deep OTM
            V = np.maximum(V, np.maximum(K - A_grid[np.newaxis, :], 0.0))

        # At fixing date n-1: A jumps to ((n-1)*A_old + S) / n
        V_new = np.zeros_like(V)
        for i in range(Nx + 1):
            A_new = ((n - 1) * A_grid + S_grid[i]) / n
            V_new[i, :] = np.interp(A_new, A_grid, V[i, :])
        V = V_new
        V = np.maximum(V, np.maximum(K - A_grid[np.newaxis, :], 0.0))

    # At t=0: A_0 = S_0 (first fixing includes t=0)
    j0 = np.argmin(np.abs(A_grid - S0))
    return np.interp(np.log(S0), x_grid, V[:, j0])

def main():
    S0, K, r, sigma, T = 1, 1.1, 0.06, 0.50, 1.0
    print(f"American-Asian put via 2D FDM:")
    print(f"{'N_fix':>6s} {'Nx':>5s} {'Na':>5s} {'FDM price':>10s}")
    for N_fix, Nx, Na in [(20, 50, 40), (50, 100, 80), (50, 150, 120)]:
        p = american_asian_put_fdm(S0, K, r, sigma, T, N_fix=N_fix, Nx=Nx, Na=Na)
        print(f"{N_fix:6d} {Nx:5d} {Na:5d} {p:10.4f}")
    print()

if __name__ == "__main__":
    main()