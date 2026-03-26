import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm
from forecasting_models import SDEDataset


class CausalPad1d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, 0))

class WeightNormTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_rate):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        
        # Branch 1: Convolutional sequence using weight_norm instead of BatchNorm
        self.pad1 = CausalPad1d(self.padding)
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)
        
        self.pad2 = CausalPad1d(self.padding)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)
        
        # Branch 2: Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        
        res = self.downsample(x)
        return self.relu2(out + res)

class WeightNormTCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels=[32, 32, 32, 32], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(WeightNormTCNBlock(in_channels, out_channels, kernel_size, dilation_size, dropout))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.network(x)
        return self.linear(y[:, :, -1]) # Predict on the last step


def TCNforecast(data, train_ratio, lookback=64, horizon=16, epochs=10, lr=0.001):
    """
    Args:
        data        : np.ndarray, shape (timesteps, num_paths)
        train_ratio : float
        lookback    : input context window
        horizon     : steps to forecast
    Returns:
        trained_model, history
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    paths = np.asarray(data)             
    paths = paths.T                          
    n = len(paths)
    split = int(n * train_ratio)
    train_paths, val_paths = paths[:split], paths[split:]

    train_ds = SDEDataset(train_paths, lookback, horizon)
    val_ds   = SDEDataset(val_paths,   lookback, horizon)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    #Model
 
    model = WeightNormTCN(
        input_size   = 1,
        output_size  = horizon,
        num_channels = [32, 32, 32, 32],
        kernel_size  = 3,
        dropout      = 0.2,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x = x.unsqueeze(1).to(device)        
            y = y.to(device)                     

            optimizer.zero_grad()
            pred = model(x)               
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_dl.dataset)

        # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x = x.unsqueeze(1).to(device)
                y = y.to(device)
                val_loss += criterion(model(x), y).item() * len(x)
        val_loss /= len(val_dl.dataset)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 2 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{epochs}  train={train_loss:.5f}  val={val_loss:.5f}")

    print("\nTCN training complete.")
    return model, history


def tcn_forecast_sequential(data, train_ratio, median, plot, lookback=64, horizon=16, epochs=10):
    """
    Wrapper that makes TCN return (forecast, test) matching classical models.
    """
    model, history = TCNforecast(data, train_ratio, lookback=lookback,
                                  horizon=horizon, epochs=epochs)

    device = next(model.parameters()).device

    if median:
        median_path = np.median(data, axis=1)
        distances   = np.mean((data - median_path[:, None])**2, axis=0)
        series = data[:, np.argmin(distances)]
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
            start = train_size + i - lookback
            if start < 0:
                start = 0
            window = norm_series[start : start + lookback]
            x_t = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(x_t).squeeze(0).cpu().numpy()
            forecast[i] = pred[0] * sigma + mu

    if plot:
        from forecasting_models import plot_generic
        plot_generic(train, forecast, test, method="TCN")
    return forecast, test
