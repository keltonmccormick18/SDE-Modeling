SDE Modeling & Forecasting:

Implements 5 stochastic differential equations — GBM, CIR, Heston, Ornstein-Uhlenbeck, and Double-Well — and benchmarks 7 forecasting methods against each, including a Temporal Convolutional Network and Transformer alongside classical ML approaches. Simulated via Euler-Maruyama discretization.
The goal: understand how model structure (stationarity, mean-reversion, stochastic volatility) affects forecasting performance across both classical and deep learning methods.
Full writeup with derivations and results: [Overleaf]([url](https://www.overleaf.com/read/zspjdtpwjmtc#ba817d)) 

Structure: sde_models.py defines the SDEs · euler_maruyama.py handles simulation · forecasting_models.py and tempconvnetwork.py implement the forecasting methods · main.py ties it together · output plots live in per-SDE folders.
Stack: Python · NumPy · pandas · TensorFlow · scikit-learn · matplotlib/seaborn
