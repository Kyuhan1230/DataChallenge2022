import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# 통계치 생성
def calc_y_metrics(y_true, y_pred):
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

