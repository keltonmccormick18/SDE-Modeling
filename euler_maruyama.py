
import numpy as np
import matplotlib.pyplot as plt

def euler_maruyama(X0, drift, diffusion, T, dt, num_paths):
    
    N = int(T/dt)
    timesteps = np.linspace(0.,T,N+1) 
    X = np.zeros((N+1, num_paths))
    X[0] = X0

    for i in range(N):
        dW = np.random.standard_normal(num_paths) * np.sqrt(dt)
        a = drift(X[i],timesteps[i])
        b = diffusion(X[i],timesteps[i])

        X[i+1] = X[i] + a * dt + b * dW

    return timesteps, X


def euler_maruyama_heston(X0, v0, mu, kappa, theta, sigma, rho, T, dt):
    
    n_steps = int(T / dt)
    t       = np.linspace(0, T, n_steps)
    X_path  = np.zeros(n_steps)
    v_path  = np.zeros(n_steps)

    X_path[0] = X0
    v_path[0] = v0

    # Cholesky factor
    L = np.array([[1.0,               0.0],
                  [rho, np.sqrt(1 - rho**2)]])

    for i in range(1, n_steps):
        Z = np.random.randn(2)

        dW = L @ Z * np.sqrt(dt) 
        v_prev = max(v_path[i-1], 0.0) 
        
        v_path[i] = (v_path[i-1]
                     + kappa * (theta - v_prev) * dt
                     + sigma * np.sqrt(v_prev) * dW[1])

        X_path[i] = (X_path[i-1]
                     + mu * X_path[i-1] * dt
                     + np.sqrt(v_prev) * X_path[i-1] * dW[0])

    return t, X_path, v_path
