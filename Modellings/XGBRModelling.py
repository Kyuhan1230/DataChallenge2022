"""
4. XGBoostRegressor
    - fit
    - _model_cv
    - _find_param
    - refit
    - predict
    - get_feature_importance
    - get_SHAP
"""
import shap
import optuna
import pandas as pd
import random as rn
import xgboost as xgb   # Python wrapper XGBoost Module
from Modellings.metrics import *
from Modellings.CrossValidation import CV_TD

rn.seed(777)
np.random.seed(777)
warnings.filterwarnings(action='ignore')


class XGBR_TD:
    def __init__(self):
        self.xgb_model = None
        self.x_columns = None
        self.x_data, self.y_data, = None, None
        self.cv, self.n_splits = None, None
        self.x_splited, self.y_splited = None, None
        self.history, self.refit_metrics = None, None
        self.param_input, self.study, self.cv_score, self.best_param = None, None, None, None

    def fit(self, x, y, param=None):
        if param is None:
            # https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster
            # Except n_estimators 100 --> 300
            param = {'n_estimators': 300, 'learning_rate': 0.3, 
                     'max_depth': 6,
                     'max_bin': 256, 'alpha': 0, 'lambda': 1}
        xgb_model = xgb.XGBRegressor(**param)
        self.history = xgb_model.fit(x, y, verbose=False)
        self.xgb_model = xgb_model

        if self.x_data is None:
            self.x_data = np.array(x)

        if type(x) == pd.core.frame.DataFrame:
            setattr(self, 'x_columns', x.columns.to_list())
        else:
            setattr(self, 'x_columns', [f'X{col + 1}' for col in range(x.shape[1])])

    def _model_cv(self, trial) -> float:
        cv = self.cv

        if self.param_input is None:
            param_range = {'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
                           'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
                           'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                           'max_depth': trial.suggest_int('max_depth', 1, 10),
                           'learning_rate': trial.suggest_uniform('learning_rate', 0.00001, 1),
                           'min_child_weight' : trial.suggest_int('min_child_weight', 1, 100),    # Overfitting 제어
                           'gamma' : trial.suggest_int('gamma', 1, 3),    # Overfitting 제어
                           'colsample_bytree' : trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1)    # 무작위성 추가
                          }
        else:
            param_range = self.param_input

        self.xgb_model = xgb.XGBRegressor(**param_range)

        k = 0
        while k < self.n_splits:
            self.x_group = list(self.x_splited.values())
            self.y_group = list(self.y_splited.values())

            x_train, x_test = self.x_group[k][0], self.x_group[k][1]
            y_train, y_test = self.y_group[k][0], self.y_group[k][1]

            print("train_data", x_train.shape, y_train.shape, "test_data", x_test.shape, y_test.shape)

            self.xgb_model.fit(x_train, y_train)

            y_pred = self.predict(x_test)

            cv._calculate_metrics(y_test, y_pred)
            k += 1

        metrics = cv._calculate_avg_metrics()

        score = metrics[self.tracking]

        return score
    
    def find_param(self, x_data, y_data, 
                   split_type="Nested", n_splits=5,
                   trial_num=10, param_range=None,
                   tracking='R2'):
        self.n_splits = n_splits
        if type(x_data) == pd.core.frame.DataFrame:
            setattr(self, 'x_columns', x_data.columns.to_list())
        else:
            setattr(self, 'x_columns', [f'X{col + 1}' for col in range(x_data.shape[1])])

        self.x_data, self.y_data = np.array(x_data), np.array(y_data)
        self.cv = CV_TD()
        self.x_splited = self.cv.split_timeseries(data=x_data, TimeSeriesSplit_=split_type)
        self.y_splited = self.cv.split_timeseries(data=y_data, TimeSeriesSplit_=split_type)
        
        if tracking is 'R2':
            direction = 'maximize'
        elif tracking is 'MSE':
            direction = 'minimize'
        else:
            direction = None
            raise TypeError("Tracking is should be one of ['R2', 'MSE', '?']")
        
        self.tracking = tracking
        
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=777), direction=direction)
        if param_range is not None:
            self.param_input = param_range
        study.optimize(self._model_cv, n_trials=trial_num)

        self.best_param = study.best_params
        self.cv_score = study.best_trial.value
        self.study = study

    def refit(self):
        x_train = self.x_data
        y_train = self.y_data

        self.xgb_model = xgb.XGBRegressor(**self.best_param)
        refit_history = self.xgb_model.fit(x_train, y_train, verbose=False)

        return refit_history

    def predict(self, x):
        y_pred = self.xgb_model.predict(x)
        return y_pred

    def get_feature_importance(self) -> pd.DataFrame:
        feature_importance = list(self.xgb_model.feature_importances_)
        col = self.x_columns
        feature_importance = pd.DataFrame(feature_importance, index=col, columns=['Feature importance'])
        return feature_importance

    def get_shap(self, x=None) -> pd.DataFrame:
        explainer = shap.Explainer(self.xgb_model)
        if x is None:
            x = self.x_data
        col = self.x_columns

        shap_values = explainer.shap_values(x)
        shap_v_means = abs(shap_values).mean(axis=0)

        feature_importance = pd.DataFrame(shap_v_means, index=col, columns=['Feature importance-SHAP'])

        return feature_importance
