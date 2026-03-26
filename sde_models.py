
import numpy as np

class OrnsteinUhlenbeck:

    #dx_t = theta (mu - X_t) dt + sigma dW_t

    #theta - speed of mean reversion
    #mu - mean
    #sigma - variance/volatility


    def __init__(self, theta, mu, sigma, X0):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.X0 = X0

    def drift(self, x, t):
        return self.theta * (self.mu - x)

    def diffusion(self, x, t):
        return self.sigma

    
class GeometricBrownianMotion:

    #dx_t = mu X_t dt + sigma X_t dW_t

    def __init__(self, mu, sigma, X0):
        self.mu = mu
        self.sigma = sigma
        self.X0 = X0

    def drift (self, x, t):
        return self.mu * x
    
    def diffusion (self, x, t):
        return self.sigma * x






class Nonlineardrift:

    #dX_t = (alpha X_t - gamma X_t**3)dt + sigma dW_t

    def __init__(self, alpha, gamma, sigma, X0):
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma
        self.X0 = X0

    def drift (self, x, t):
        return self.alpha * x - self.gamma * (x**3)
    
    def diffusion (self, x, t):
        return self.sigma

class Cox_Ingersoll_Ross:

    #dX_t = theta (mu - X_t) dt + sigma * (X_t ** gamma (gamma = 1/2) ) dW_t

    def __init__(self, theta, mu, sigma, gamma, X0):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.gamma = gamma
        self.X0 = X0
    
    def drift (self, x, t):
        return self.theta * (self.mu - x)
    
    def diffusion(self, x, t):
        return self.sigma * (x ** self.gamma)

    
class HestonModel:
    # dX_t = mu * X_t * dt + sqrt(v_t) * X_t * dW^1_t
    # dv_t = kappa * (theta - v_t) * dt + sigma * sqrt(v_t) * dW^2_t
    # corr(dW^1, dW^2) = rho

    def __init__(self, mu, kappa, theta, sigma, rho, v0, X0):
        self.mu    = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho   = rho
        self.v0    = v0
        self.X0    = X0

    def simulate(self, num_paths=1, T=100, dt=0.01):
        n_steps = int(T / dt)
        t       = np.linspace(0, T, n_steps)

        X_paths = np.zeros((n_steps, num_paths))
        v_paths = np.zeros((n_steps, num_paths))

        X_paths[0] = self.X0
        v_paths[0] = self.v0

        L = np.array([[1.0,                          0.0],
                      [self.rho, np.sqrt(1 - self.rho**2)]])

        for i in range(1, n_steps):
            Z    = np.random.randn(2, num_paths)
            dW   = L @ Z * np.sqrt(dt)

            v_prev = np.maximum(v_paths[i-1], 0.0)

            v_paths[i] = (v_paths[i-1]
                          + self.kappa * (self.theta - v_prev) * dt
                          + self.sigma * np.sqrt(v_prev) * dW[1])

            X_paths[i] = (X_paths[i-1]
                          + self.mu * X_paths[i-1] * dt
                          + np.sqrt(v_prev) * X_paths[i-1] * dW[0])

        return t, X_paths, v_paths
