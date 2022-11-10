"""
4. XGBoostRegressor, LightGBMRegressor
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
import lightgbm as lgbm
from Modellings.metrics import *
from Modellings.CrossValidation import CV_TD

rn.seed(777)
np.random.seed(777)
warnings.filterwarnings(action='ignore')


class LGBMR_TD:
    def __init__(self):
        self.lgbm_model = None
        self.x_columns = None
        self.x_data, self.y_data, = None, None
        self.cv, self.n_splits = None, None
        self.x_splited, self.y_splited = None, None
        self.history, self.refit_metrics = None, None
        self.param_input, self.study, self.cv_score, self.best_param = None, None, None, None

    def fit(self, x, y, param=None):
        # n_estimators 만 제외하고 나머지는 기본 값 사용
        if param is None:
            param = {'n_estimators': 300}
            lgbm_model = lgbm.LGBMRegressor(**param)
        else:
            lgbm_model = lgbm.LGBMRegressor()
#         lgbm_model = lgbm.LGBMRegressor(**param)
        # 초창기 사용은 reg_lambda=55, reg_alpha=60 였었음.
        self.history = lgbm_model.fit(x, y, verbose=False)
        self.lgbm_model = lgbm_model

        if self.x_data is None:
            self.x_data = np.array(x)

        if type(x) == pd.core.frame.DataFrame:
            setattr(self, 'x_columns', x.columns.to_list())
        else:
            setattr(self, 'x_columns', [f'X{col + 1}' for col in range(x.shape[1])])

    def _model_cv(self, trial) -> float:

        cv = self.cv

        if self.param_input is None:
            param_range = {"n_estimators": trial.suggest_int('n_estimators', 100, 500),
                           "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                           "num_leaves": trial.suggest_int("num_leaves", 5, 30, step=20),
                           "max_depth": trial.suggest_int("max_depth", 3, 12),
                           "min_child_samples": trial.suggest_int("min_child_samples", 10, 500, step=100),
                           "max_bin": trial.suggest_int("max_bin", 200, 300),
                           "reg_alpha": trial.suggest_int("reg_alpha", 0, 100, step=5),
                           "reg_lambda": trial.suggest_int("reg_lambda", 0, 100, step=5),}
        else:
            param_range = self.param_input

        self.lgbm_model = lgbm.LGBMRegressor(**param_range)

        k = 0
        while k < self.n_splits:
            self.x_group = list(self.x_splited.values())
            self.y_group = list(self.y_splited.values())

            x_train, x_test = self.x_group[k][0], self.x_group[k][1]
            y_train, y_test = self.y_group[k][0], self.y_group[k][1]

            print("train_data", x_train.shape, y_train.shape, "test_data", x_test.shape, y_test.shape)

            self.lgbm_model.fit(x_train, y_train)

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

        self.lgbm_model = lgbm.LGBMRegressor(**self.best_param)
        refit_history = self.lgbm_model.fit(x_train, y_train, verbose=False)

        return refit_history

    def predict(self, x):
        y_predict = self.lgbm_model.predict(x)
        return y_predict

    def get_feature_importance(self) -> pd.DataFrame:
        feature_importance = list(self.lgbm_model.feature_importances_)
        col = self.x_columns
        feature_importance = pd.DataFrame(feature_importance, index=col, columns=['Feature importance'])
        return feature_importance

    def get_shap(self, x=None) -> pd.DataFrame:
        explainer = shap.Explainer(self.lgbm_model)
        if x is None:
            x = self.x_data
        col = self.x_columns

        shap_values = explainer.shap_values(x)
        shap_v_means = abs(shap_values).mean(axis=0)

        feature_importance = pd.DataFrame(shap_v_means, index=col, columns=['Feature importance-SHAP'])

        return feature_importance
