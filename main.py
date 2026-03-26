
import numpy as np
import matplotlib.pyplot as plt
from sde_models import OrnsteinUhlenbeck
from sde_models import GeometricBrownianMotion
from sde_models import Nonlineardrift
from sde_models import Cox_Ingersoll_Ross
from sde_models import HestonModel
from euler_maruyama import euler_maruyama
from forecasting_models import arima_forecast
from forecasting_models import NonLinear
from forecasting_models import RandomForest
from forecasting_models import GradientBoosting
from forecasting_models import LSTMforecast
from forecasting_models import transformer_forecast
from tempconvnetwork import tcn_forecast_sequential

def simulationplot(t,paths, num_paths,sde_name:str):
    plt.figure(figsize=(10, 6))
    for i in range(min(num_paths,10)):
        plt.plot(t, paths[:, i], label=f'Path {i+1}')

    plt.title(f'{sde_name} Process Simulation using Euler-Maruyama')
    plt.xlabel('Time (t)')
    plt.ylabel('X(t)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/Users/kelton/stochasproject/plots/{sde_name}_process_sim.png', dpi=150, bbox_inches="tight")
    plt.show()



ForecastModels = {
    1 : arima_forecast,
    2 : NonLinear,
    3 : RandomForest,
    4 : GradientBoosting,
    5 : LSTMforecast,
    6 : tcn_forecast_sequential,
    7 : transformer_forecast
}

ForecastModelNames = {
    1 : "ARIMA",
    2 : "Non-Linear",
    3 : "Random Forest",
    4 : "Gradient Boosting",
    5 : "LSTM",
    6 : "TCN",
    7 : "Transformer"
}
SDEModels = {
    1 : OrnsteinUhlenbeck,
    2 : GeometricBrownianMotion,
    3 : Nonlineardrift,
    4 : Cox_Ingersoll_Ross,
    5 : HestonModel 
}

SDEMODELPARAMS = {
    1: [0.7,1.5,0.6,0.0],
    2: [0.005, 0.3, 10],
    3: [1.0, 1.0, 0.5, 1.0],
    4: [1.0, 1.5, 0.5, 0.5, 0.5],
    5: [0.05, 2.0, 0.04, 0.3, -0.7, 0.04, 5.0]
}

SDEModelNames = {
    1 : "Ornstein Uhlenbeck", #OrnsteinUhlenbeck
    2 : "Geometric Brownian Motion", #GeometricBrownianMotion
    3 : "Double Well", #Nonlineardrift
    4 : "Cox-Ingersoll-Ross", #Cox_Ingersoll_Ross
    5 : "Heston Model", #HestonModel       
}


def compute_metrics(forecast, test):

    mse  = np.mean((forecast - test) ** 2)
    mae  = np.mean(np.abs(forecast - test))
    rmse = np.sqrt(mse)

    actual_dir = np.sign(np.diff(test))
    pred_dir   = np.sign(np.diff(forecast))
    dir_acc    = np.mean(actual_dir == pred_dir)

    mask = np.abs(test) > 1e-8
    mape = np.mean(np.abs((forecast[mask] - test[mask]) / test[mask])) * 100
    bias = np.mean(forecast - test)
    mean_error = np.abs(forecast.mean() - test.mean())
    var_ratio = np.var(forecast) / (np.var(test) + 1e-8)

    return {
        "mse"        : float(mse),
        "mae"        : float(mae),
        "rmse"       : float(rmse),
        "dir_acc"    : float(dir_acc),
        "mape"       : float(mape),
        "bias"       : float(bias),
        "mean_error" : float(mean_error),
        "var_ratio"  : float(var_ratio),
    }



def representative_path(data):
    #path closest to median for data with multiple paths
    median_path = np.median(data, axis=1)
    distances   = np.mean((data - median_path[:, None])**2, axis=0)
    return data[:, np.argmin(distances)]



def CompareForecastModels(sde_id:int):
    num_paths = 30
    plot = True
    T = 100
    dt = 0.01 
    params = SDEMODELPARAMS[sde_id]
    sde = SDEModels[sde_id](*params)
    if sde_id == 5: #HESTON SPECIAL CASE
        t, paths, _ = sde.simulate(num_paths=num_paths, T = T, dt = dt)
    else:
        t, paths = euler_maruyama(sde.X0, sde.drift, sde.diffusion, T = T, dt = dt, num_paths = num_paths)
    simulationplot(t,paths, num_paths, sde_name = SDEModelNames[sde_id])
    for i in ForecastModelNames:
        if i in range(1,6):
            if (sde_id == 3 or sde_id == 5): #special case for Double well, Heston, as taking mean destroys the structure.
                forecast, test = ForecastModels[i](representative_path(paths), 0.8, plot)
            else:
                forecast, test = ForecastModels[i](paths.mean(axis = 1), 0.8, plot)
        else:
            if (sde_id == 3 or sde_id == 5):
                forecast, test = ForecastModels[i](paths, 0.8, True, plot)
            else:
                forecast, test = ForecastModels[i](paths,0.8,False, plot)
        print(compute_metrics(forecast, test))


CompareForecastModels(1)
