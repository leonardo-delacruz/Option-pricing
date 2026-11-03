# Geometric Brownian motion is defined as dW=  dt Z
# where Z is a normal random variable
import matplotlib.pyplot as plt


def random_number_generator():
    import numpy as np
    rng=np.random.default_rng()
    return rng.standard_normal()

def brownian_motion(T=1, n_steps=1, m_paths=1, seed=None):
    import numpy as np
    rng = np.random.default_rng(seed)
    dt=T/n_steps
    dW=rng.standard_normal(size=(m_paths,n_steps))*np.sqrt(dt)
    W=np.cumsum(dW,axis=1)
    W=np.concatenate([np.zeros((m_paths,1)),W],axis=1)
    return W

def geometric_brownian_motion(S0=100.0,drift=0.1, volatility=0.5, T=1, n_steps=1, m_paths=1, seed=None):
    import numpy as np
    rng=np.random.default_rng(seed)
    dt=T/n_steps
    dW=volatility*rng.standard_normal(size=(m_paths,n_steps))*np.sqrt(dt)
    W=np.cumsum(dW,axis=1)
    W=np.concatenate([np.zeros((m_paths,1)),W],axis=1)
    t=np.linspace(0,T,n_steps+1)
    S=S0*np.exp((drift-0.5*volatility**2)*t+volatility*W)
    return t, S

def discretized_geometric_brownian_motion(S0=100.0,drift=0.1, volatility=0.5, T=1, n_steps=1, m_paths=1, seed=None):
    import numpy as np
    rng=np.random.default_rng(seed)
    dt=T/n_steps
    dW=rng.standard_normal(size=(m_paths,n_steps))*np.sqrt(dt)
    factors=1+drift*dt+volatility*dW
    paths=S0*np.cumprod(factors, axis=1)
    t = np.linspace(0, T, n_steps + 1)
    S=np.concatenate([np.full((m_paths, 1), S0), paths],axis=1)
    return t, S

def generate_GBM_paths(S0=1,
                       rate=0.06, volatility=0.5, m_paths=2, t_exercise=1,
                       n_steps=252, seed=918723098,
                       antithetic=True):
    """ Function to generate random paths, requires numpy to run. Uses exact solution  """
    import numpy as np
    rng = np.random.default_rng(seed)
    dt = t_exercise / n_steps
    import numpy as np
    if antithetic:
        half = m_paths // 2
        Ns = rng.standard_normal((half, n_steps))
        Ns = np.concatenate([Ns, -Ns], axis=0)
    else:
        Ns = rng.standard_normal(size=(m_paths, n_steps))
    # Exact solution for GBM
    exp_factor = np.exp((rate - 0.5 * volatility ** 2) * dt + volatility * Ns * np.sqrt(dt))
    S_iterated = S0 * np.cumprod(exp_factor, axis=1)
    S = np.concatenate([np.full((m_paths, 1), S0), S_iterated], axis=1)
    return S


def main():
    import matplotlib.pyplot as plt
    m_paths = 20
    times = [i * 1 / 252 for i in range(253)]
    t, S =discretized_geometric_brownian_motion(S0=100,drift=0.08, volatility=0.3, T=1,
                                    n_steps=252, m_paths=m_paths, seed=120910923091823)

    for path in S:
        plt.plot(t,path)
    plt.show()


if __name__ == "__main__":
    main()