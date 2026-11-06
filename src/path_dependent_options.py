from geometric_brownian_motion import generate_GBM_paths
import numpy as np

# We leave this as free function
def regression_function(x, deg):
    return np.vstack([x ** k for k in range(deg + 1)]).T

def american_put_LSM(S0=1, strike_price=1.1, rate=0.06, volatility=0.5, m_paths=8, t_exercise=3,
                                 n_steps=3, seed=91873098, deg=2):
    """
Amrican put by Least Squares Monte Carlo
 Parameters:
        S0 (float): Stock price at initial time
        strike_price (float): Strike price
        rate (float): risk-free rate
        volatility (float): volatility
        m_paths (int): number of paths to simulate
        t_exercise (int): exercise date ( one year default)
        n_steps (int): number of steps
        seed (int): random seed
        deg (int): degree of the polynomial we use to regress
    Returns: price and standard error
    """
    import numpy as np
    dt = t_exercise/n_steps
    #We use always antithetic variates
    S = generate_GBM_paths(
        S0=S0,
        rate=rate,
        volatility=volatility,
        m_paths=m_paths,
        t_exercise=t_exercise,
        n_steps=n_steps,
        seed=seed,
        antithetic=True,
    )
    # Compare prices at once at time=t_exercise
    payoff=np.maximum(strike_price-S[:,-1], 0.0)
    #create auxiliary array to fill for decision making
    decisions=np.zeros_like(S, dtype=bool)
    #decision at t_exercise is trivial
    decisions[:,-1]=payoff>0

    #start of the iteration in decreasing steps
    for n in range(n_steps-1, 0,-1):
        payoff=payoff*np.exp(-rate*dt)
        in_the_money=np.where(S[:,n]<strike_price)[0]
        if len(in_the_money)==0:
            continue
        #Regression step
        X=regression_function(S[in_the_money, n], deg)
        Y=payoff[in_the_money]
        coeff, *_=np.linalg.lstsq(X,Y,rcond=None)
        continuation=X @ coeff
        #decisions at each step
        exercise_now=np.maximum(strike_price-S[in_the_money, n], 0.0)
        exercise_idx=in_the_money[continuation < exercise_now]
        payoff[exercise_idx]=exercise_now[continuation<exercise_now]
        #update decision table
        decisions[exercise_idx, :]=False
        decisions[exercise_idx, n]=True
    #We discount the value from t_1 to 0
    price= np.exp(-rate*dt)*payoff.mean()
    #We have to compute the standard error
    halfpaths = m_paths//2  # number of PAIRS
    pair_avg = (payoff[:halfpaths] + payoff[halfpaths:]) / 2.0  # length M
    std_error = pair_avg.std(ddof=1) / np.sqrt(halfpaths)  # antithetic SE

    return price, std_error


# Arithmetic Asian
def arithmethic_asian(S0=1, rate=0.06, volatility=0.5, m_paths=150, t_exercise=1, n_steps=252, seed=91873098):
    import numpy as np
    S = generate_GBM_paths(S0=S0, rate=rate, volatility=volatility, m_paths=m_paths,
                           t_exercise=t_exercise, n_steps=n_steps, seed=seed,
                           # We always use antithetic variables
                           antithetic=True)
    # Aritmethic average
    A_T = S.mean(axis=1)

    # pricing
    payoff = np.maximum(S[:, -1] - A_T, 0.0)
    price_asian = np.exp(-rate * t_exercise) * payoff.mean()
    return price_asian


def americanized_asian(S0=1, strike_price=1.1, rate=0.06, volatility=0.5, m_paths=150, t_exercise=1, n_steps=252, seed=91873098,
                       deg=None):
    """ American Asian put by Least Squares Monte Carlo
    Parameters:
        S0 (float): Stock price at initial time
        strike_price (float): Strike price
        rate (float): risk-free rate
        volatility (float): volatility
        m_paths (int): number of paths to simulate
        t_exercise (int): exercise date ( one year default)
        n_steps (int): number of steps
        seed (int): random seed
        deg (int): here we use by default degree of 2 for simplicity
    Returns: price and standard error
    """
    # Regression function is now a polynomial basis in two variables: here is where the strong path dependence enters
    # For simplicity I will assume only a second order approximation
    dt = t_exercise / n_steps
    S = generate_GBM_paths(
        S0=S0,
        rate=rate,
        volatility=volatility,
        m_paths=m_paths,
        t_exercise=t_exercise,
        n_steps=n_steps,
        seed=seed,
        antithetic=True,
    )
    def regression_2_vars(x, a, deg=None):
        return np.vstack([np.ones_like(x), x, x ** 2, a, a ** 2, x * a]).T

    # Running arithmetic average A_{t_n} = (1/(n+1)) * sum_{k=0..n} S_{t_k}
    # (includes t_0 = S_0 in the average)
    cumsum_S = np.cumsum(S, axis=1)
    A = cumsum_S / np.arange(1, n_steps + 2)  # (m_paths, n_steps+1)

    # Exercise payoff at each t_n: max(K - A_{t_n}, 0) for a put on the average
    exercise_payoff = np.maximum(strike_price - A, 0.0)  # (m_paths, n_steps+1)

    # Backward induction: starting payoff is the one at t_{n_steps}
    payoff = exercise_payoff[:, -1].copy()

    # backward in time
    for n in range(n_steps - 1, 0, -1):
        payoff = payoff * np.exp(-rate * dt)
        in_the_money = np.where(A[:, n] < strike_price)[0]  # ITM for a put: A < strike_price
        if len(in_the_money) == 0:
            continue
        X = regression_2_vars(S[in_the_money, n], A[in_the_money, n])
        Y = payoff[in_the_money]
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        continuation = X @ coef
        ex_now = exercise_payoff[in_the_money, n]
        ex_mask = continuation < ex_now
        payoff[in_the_money[ex_mask]] = ex_now[ex_mask]

    price_am_asian = payoff.mean() * np.exp(-rate * dt)
    # We have to compute the standard error
    halfpaths = m_paths // 2  # number of PAIRS
    pair_avg = (payoff[:halfpaths] + payoff[halfpaths:]) / 2.0  # length M
    std_error = pair_avg.std(ddof=1) / np.sqrt(halfpaths)  # antithetic SE
    return price_am_asian, std_error

