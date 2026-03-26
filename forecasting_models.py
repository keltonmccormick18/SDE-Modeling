from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
import bisect
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", ConvergenceWarning)


def lagged_data(data, numlags):
        X, y = [], []
        for i in range(numlags, len(data)):
            X.append(data[i-numlags:i])
            y.append(data[i])
        return X, np.array(y)

def split_sequence(data,steps):
        X,y = [],[]
        for i in range(len(data)):
            endi = i+steps
            if endi > (len(data)-1):
                break
            xi = data[i:endi]
            yi = data[endi]
            X.append(xi)
            y.append(yi)
        return X,y

def plot_generic(train_data, forecast, test_data, method):
    t_train = np.arange(len(train_data))
    t_test  = np.arange(len(train_data), len(train_data) + len(test_data))

    plt.figure(figsize=(12, 4))
    plt.plot(t_train, train_data, label="Train",        color="#4C72B0", lw=1.5)
    plt.plot(t_test,  test_data,  label="Ground truth", color="#55A868", lw=1.5)
    plt.plot(t_test,  forecast,   label="Forecast",     color="#C44E52", lw=1.5, linestyle="--")
    plt.axvline(len(train_data), color="gray", lw=0.8, linestyle=":")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.title(f"Forecast vs Ground Truth with {method} for Heston Process")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'/Users/kelton/stochasproject/plots/{method}_forecast.png', dpi=150, bbox_inches="tight")
    plt.show()



#####

def arima_forecast(data, train_ratio, plot):
    max_p = 4
    max_d = 2
    max_q = 4
    n = len(data)
    train_size = int(train_ratio*n)
    train = data[:train_size]
    test = data[train_size:] # segment the data into training and testing

    best_aic = np.inf
    best_order = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0 and d == 0:
                    continue
                try:
                    model = ARIMA(train, order=(p, d, q))
                    model_fit = model.fit(method_kwargs={"warn_convergence": False})

                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_order = (p, d, q)

                except Exception as e:
                    print(f"ARIMA({p},{d},{q}) failed: {e}")
                    continue

    best_model = ARIMA(train, order = best_order)
    best_model_fit = best_model.fit()
    forecast = best_model_fit.forecast(steps = len(test))

    if plot:
        plot_generic(train, forecast, test, method = "ARIMA")
    return forecast, test



def NonLinear(data, train_ratio, plot):

    data = data.flatten()
    n = len(data)
    train_size = int(train_ratio*n)

    test = data[train_size:]
    train = data[:train_size]

    x_t = train[:-1]
    x_tp1 = train[1:]

    X = np.column_stack([x_t, x_t**2, x_t**3])

    reg = LinearRegression(fit_intercept=False)
    reg.fit(X, x_tp1)

    a_0, b_0, c_0 = reg.coef_

    def model(x_t, a, b, c):
        return a*x_t + b*x_t**2 + c*x_t**3

    params, _ = curve_fit(model, x_t, x_tp1, p0=[a_0, b_0, c_0])
    a, b, c = params

    def forecast_nonlinear(xinit, params, steps):
        forecast = np.zeros(steps)
        x = xinit

        for i in range(steps):
            x = a*x + b*x**2 + c*x**3
            forecast[i] = x
        
        return forecast
    
    forecast = forecast_nonlinear(data[train_size],params,(n-train_size))

    if plot:
        plot_generic(train, forecast, test, method = "Nonlinear")
    return forecast, test


def RandomForest(data, train_ratio, plot):

    data = data.flatten()
    X, y = lagged_data(data, 3)

    n = len(y)
    train_size = int(n*train_ratio)

    train_data = y[:train_size]
    train_X = X[:train_size]
    test_data = y[train_size:]
    test_X = X[train_size:]

    rf_regressor = RandomForestRegressor(n_estimators = 10)
    rf_regressor.fit(train_X,train_data)
    
    forecast = rf_regressor.predict(test_X)

    if plot:
        plot_generic(train_data, forecast, test_data, method = "Random Forest")
    return forecast, test_data

def GradientBoosting(data, train_ratio, plot):

    data = data.flatten()
    n = len(data)
    train_size = int(n*train_ratio)
    X, y = lagged_data(data, 3)

    train_data = y[:train_size]
    train_X = X[:train_size]
    test_data = y[train_size:]
    test_X = X[train_size:]

    gbcregressor = GradientBoostingRegressor()

    gbcregressor.fit(train_X, train_data)

    forecast = gbcregressor.predict(test_X)

    if plot:
        plot_generic(train_data, forecast, test_data, method = "Gradient Boosting")
    return forecast, test_data
    


def LSTMforecast(data, train_ratio, plot):
    steps = 10
    X, y = split_sequence(data, steps)

    n = len(X)
    train_size = int(n * train_ratio)

    train_X    = np.array(X[:train_size]).reshape(-1, steps, 1)
    test_X     = np.array(X[train_size:]).reshape(-1, steps, 1)
    train_data = np.array(y[:train_size])
    test_data  = np.array(y[train_size:])

    model = Sequential([
        LSTM(200, input_shape=(steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_X, train_data, epochs=15, verbose=0)

    forecast = model.predict(test_X).flatten()

    if plot:
        plot_generic(train_data, forecast, test_data, method="LSTM")

    return forecast, test_data

#TRANSFORMER:::






def make_dataloaders(data, train_ratio, cfg):
    paths = data.T
    n = len(paths)
  
    train_size = int(n*train_ratio)
    train_paths, val_paths = paths[:train_size], paths[train_size:]
    train_ds = SDEDataset(train_paths, cfg.lookback, cfg.horizon)
    val_ds = SDEDataset(val_paths, cfg.lookback, cfg.horizon)
    train_dl = DataLoader(train_ds, batch_size = cfg.batch_size, shuffle = True, pin_memory = False, num_workers = 0)
    val_dl = DataLoader(val_ds, batch_size = cfg.batch_size, shuffle = False, pin_memory = False, num_workers = 0)
    return train_dl, val_dl



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1)])


@dataclass
class Config:

    lookback: int    = 64    
    horizon: int     = 16    
 
    d_model: int     = 64     # embedding dimension (must be divisible by nhead)
    nhead: int       = 4      # number of attention heads
    num_layers: int  = 3      # number of TransformerEncoder layers
    dim_ff: int      = 256    # feed-forward hidden size
    dropout: float   = 0.1
 
    batch_size: int  = 256
    epochs: int      = 5
    lr: float        = 3e-4
    grad_clip: float = 1.0    # max gradient norm 
    val_split: float = 0.15 
 
    device: str     = "mps" 

class SDEDataset(Dataset):
    def __init__(self, paths: np.ndarray, lookback: int, horizon: int,
                 normalize: bool = True):
        self.lookback = lookback
        self.horizon  = horizon
        self.window   = lookback + horizon

        processed = []
        for path in paths:
            if normalize:
                mu, sigma = path.mean(), path.std() + 1e-8
                path = (path - mu) / sigma
            processed.append(path)

        self.paths = [torch.tensor(p, dtype=torch.float32) for p in processed]

        self.windows_per_path = [max(0, len(p) - self.window + 1) for p in self.paths]
        self.cumulative = torch.tensor(self.windows_per_path).cumsum(0).tolist()


    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, idx):
        path_idx = bisect.bisect_right(self.cumulative, idx)
        offset = idx - (self.cumulative[path_idx - 1] if path_idx > 0 else 0)
        path = self.paths[path_idx]
        x = path[offset : offset + self.lookback]
        y = path[offset + self.lookback: offset + self.window]
        return x, y
 
 
class SDETransformer(nn.Module):
    """
    Encoder-only Transformer for multi-step time-series forecasting.
 
    Pipeline:
      (batch, L, 1)
        → Linear projection  → (batch, L, d_model)
        → Positional encoding
        → N x TransformerEncoderLayer  [causal mask]
        → Take last token repr         → (batch, d_model)
        → MLP head                     → (batch, H)
    """
 
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(1, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.dropout)
 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.d_model,
            nhead           = cfg.nhead,
            dim_feedforward = cfg.dim_ff,
            dropout         = cfg.dropout,
            batch_first     = True,   
            norm_first      = True, 
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
 
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.horizon),
        )
 
        self._init_weights()
 
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
 
    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        #time series -- need to mask future data
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, lookback)  — normalised scalar time series
        Returns:
            out : (batch, horizon) — predicted future values
        """
        x = self.input_proj(x.unsqueeze(-1))
        x = self.pos_enc(x)
 
        mask = self._causal_mask(x.size(1), x.device)
        x = self.encoder(x, mask=mask)
 
        out = self.head(x[:, -1, :])
        return out
    
def transformertrain(paths: np.ndarray, train_ratio, cfg: Config = None):
    """
    Full training entry point.
 
    Args:
        paths : np.ndarray, shape (timesteps, num_paths) — your simulator output
        cfg   : Config dataclass; defaults used if None
 
    Returns:
        model   : trained SDETransformer (on cfg.device)
        history : dict with keys 'train_loss' and 'val_loss' (lists)
    """
    if cfg is None:
        cfg = Config()
 
    print(f"Device : {cfg.device}")
    print(f"Config : lookback={cfg.lookback}, horizon={cfg.horizon}, "
          f"d_model={cfg.d_model}, layers={cfg.num_layers}, heads={cfg.nhead}\n")
 
    train_dl, val_dl = make_dataloaders(paths, train_ratio, cfg)
 
    model     = SDETransformer(cfg).to(cfg.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
    )
 
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}\n")
 
    history = {"train_loss": [], "val_loss": []}
 
    for epoch in range(1, cfg.epochs + 1):
        
        # train
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item() * len(x)
 
        train_loss /= len(train_dl.dataset)
 
        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(cfg.device), y.to(cfg.device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * len(x)
        val_loss /= len(val_dl.dataset)
 
        scheduler.step()
 
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
 
        if epoch % 5 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:>3}/{cfg.epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  lr={lr_now:.2e}")
 
    print("\nTraining complete.")
    return model, history


def transformer_forecast(data, train_ratio, median, plot):
    """
    Wrapper that makes the transformer return (forecast, test)
    in the same format as classical models.
    Uses a rolling one-shot forecast over the test period.
    """
    cfg = Config()
    model, history = transformertrain(data, train_ratio, cfg)

    if median:
        median_path = np.median(data, axis=1)
        distances   = np.mean((data - median_path[:, None])**2, axis=0)
        series = data[:, np.argmin(distances)]
    # # MOVE TO OG FUNCTION similar to others.
    else:
        series = np.asarray(data).mean(axis=1)
    n = len(series)
    train_size = int(n * train_ratio)
    train = series[:train_size]
    test  = series[train_size:]

    mu, sigma = train.mean(), train.std() + 1e-8
    norm_series = (series - mu) / sigma

    forecast = np.zeros(len(test))
    model.eval()
    with torch.no_grad():
        for i in range(len(test)):
            start = train_size + i - cfg.lookback
            if start < 0:
                start = 0
            window = norm_series[start : start + cfg.lookback]
            x_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(cfg.device)
            pred = model(x_t).squeeze(0).cpu().numpy()
            forecast[i] = pred[0] * sigma + mu  
    if plot:
        plot_generic(train, forecast, test, method="Transformer")

    return forecast, test
