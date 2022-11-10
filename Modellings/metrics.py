"""
Calculate Metrics
- Y Predict Model : R2Y, RA, MSE, Y_Avg, Y_Std, |Error|avg, Error std
- X Predict Model : R2X, MSE
"""
import warnings
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings(action='ignore')


class ModelScore:
    def __init__(self):
        pass

    def calc_y_metrics(self, y_true, y_pred):
        # convert data type
        if type(y_true) != type(y_pred):
            y_true = np.array(y_true)

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        y_true = y_true.astype(np.float64)
        y_pred = y_pred.astype(np.float64)

        # calc R2 Score
        weight = np.polyfit(y_true, y_pred, 1)
        trend = np.poly1d(weight)
        r2 = round(float(r2_score(y_pred, trend(y_true))), 3)
        # calc Y stat
        ystd = round(float(y_true.std()), 3)
        yavg = round(float(y_true.mean()), 3)
        # calc RA
        avg_yerr = round(float(abs(y_true - y_pred).mean()), 5)
        if yavg != 0:
            ra = round((1 - avg_yerr / yavg), 3)
        else:
            ra = round((1 - avg_yerr), 3)
        # calc mse
        mse = round(mean_squared_error(y_true, y_pred), 5)
        # calc y_std
        std_yerr = round(float(abs(y_true - y_pred).std()), 3)

        score = {"R2": r2, "RA": ra, "MSE": mse,
                 "Y_Avg": yavg, "Y_Std": ystd, "Error_avg": avg_yerr, "Error_std": std_yerr}

        return score

    def calc_x_metrics(self, x_true, x_pred):
        x_true = np.array(x_true)
        x_pred = np.array(x_pred)
        x_num = x_true.shape[1]
        result = np.zeros(shape=(1, 7))

        for k in range(x_num):
            meas, pred = x_true[:, k], x_pred[:, k]
            metrics_k = self.calc_y_metrics(y_true=meas, y_pred=pred)
            values = np.array([list(metrics_k.values())])
            result = np.concatenate((result, values), axis=0)

        r2_mean = np.mean(result[:, 0])
        ra_mean = np.mean(result[:, 1])
        mse_mean = np.mean(result[:, 2])

        score = {"R2": r2_mean, "RA": ra_mean, "MSE": mse_mean,
                 "Y_Avg": 0, "Y_Std": 0, "Error_avg": 0, "Error_std": 0}
        return score

    def calc_adjusted_r2(self, r2: float, n: int, k: int) -> float:
        numerator = (1 - r2) * (n - 1)
        denominator = (n - k - 1)
        adjusted_r2 = float(1 - numerator / denominator)
        return adjusted_r2
    
    def calc_mean_errors(self, y_true, y_pred):
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import mean_squared_log_error
        """
        1. MSE (Mean Squared Error): 
        2. RMSE (Root Mean Squared Error)
        3. MSLE (Mean Squared Log Error)
        4. MAE (Mean Absolute Error)
        5. MAPE (Mean Absolute Percentage Error)
        6. MPE (Mean Percentage Error)
        """
        labels = ["MSE", "RMSE", "MSLE", "MAE", "MAPE", "MPE"]
        
        y_true = np.array(y_true)
        y_test = np.array(y_pred)
        
        mse =  mean_squared_error(y_true, y_pred)
        rmse =  np.sqrt(mse)
        msle =  mean_squared_log_error(y_true, y_pred)
        mae =  mean_absolute_error(y_true, y_pred)
        mape =  np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mpe = np.mean((y_true - y_pred) / y_true) * 100
        
        results = {"MSE": mse,  "RMSE": rmse,  "MSE": msle,  "MAE": mae,  "MAPE": mape,  "MPE":mpe}
        return results
