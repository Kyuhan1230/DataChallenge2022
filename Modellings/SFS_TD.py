import copy
import numpy as np
from .PLSModelling import PLS_TD
from .LGBMRModelling import LGBMR_TD
from .XGBRModelling import XGBR_TD
from .metrics import ModelScore


def _calculate_metrics(test_y, test_pred):
    model_score = ModelScore()
    result = model_score.calc_y_metrics(y_true=np.array(test_y), y_pred=test_pred)
    return result


def _evaluate_model(metrics_test, metrics_ref):
    r2_cond = float(metrics_test['R2']) > metrics_ref['R2'] * 0.99
    ra_cond = float(metrics_test['RA']) > metrics_ref['RA'] * 0.99
    mse_cond = float(metrics_test['MSE']) < metrics_ref['MSE'] * 1.1
    error_std_cond = float(metrics_test['Error_std']) < metrics_ref['Error_std'] * 1.1
    conditions = [ra_cond, mse_cond, error_std_cond]
    if r2_cond is True and conditions.count(True) >= 1:
        return True
    else:
        return False


class SFS_TD:
    """Sequential Feature Selection"""

    def __init__(self, model_type='PLS'):
        self.model_type = model_type
        self.model = None
        self.selected_columns = None
        self.last_model_score = None
        self.selected = []
        self.model_inputs = []
        self.best_params = None
        model_types = ["PLS", "LightGBM", "XGBoost"]

        if model_type in model_types:
            pass
        else:
            raise TypeError("The model type Shouldbe one of [PLS, LightGBM, XGBoost]")


    def _set_score_ref(self, ref):
        """R2_ref >= R2 * 0.99, RA_ref >= RA * 0.99,
           MSE_ref <= MSE * 1.1, Error_std_ref <= Error_std * 1.01"""
        r2_ref = round(float(ref['R2'] * 0.99), 3)
        ra_ref = round(float(ref['RA'] * 0.99), 3)
        mse_ref = round(float(ref['MSE'] * 1.01), 3)
        error_std_ref = round(float(ref['Error_std'] * 1.01), 3)
        self.score_ref = {"R2": r2_ref, "RA": ra_ref, "MSE": mse_ref, "Error_std": error_std_ref}

    def _build_init_model(self, x_total_train, y_train, x_total_test, y_test):
        """Build The Initial Model
           1. Fit Initial Model using whole variables
           2. Calculate Feature Importance(VIP or SHAP)
           3. Calculate Metrics for Set Value(R2, RA, MSE)
           4. Set Variables for 'Forward Elimination' """
        model, importance, importance_name = None, None, None
        if self.model_type == "PLS":
            # PLS 모델링
            pls = PLS_TD()
            pls.fit(x_total_train, y_train)
            # 중요도 계산
            importance = pls.calculate_vip()
            importance_name = 'VIP value'
            model = pls
        elif self.model_type == "LightGBM":
            # LightGBM 모델링
            lgbmr = LGBMR_TD()
            lgbmr.fit(x_total_train, y_train)
            # lgbmr.refit()
            # 중요도 계산
            importance = lgbmr.get_shap()
            importance_name = 'Feature importance-SHAP'
            model = lgbmr
        elif self.model_type == "XGBoost":
            # XGBoost 모델링
            xgbr = XGBR_TD()
            xgbr.find_param(x_data=x_total_train, y_data=y_train, trial_num=10)
            xgbr.refit()
            # 중요도 계산
            importance = xgbr.get_shap()
            importance_name = 'Feature importance-SHAP'
            model = xgbr
        else:
            raise TypeError("The model type Shouldbe one of [PLS, LightGBM, XGBoost]")
        # 예측값 계산
        test_pred = model.predict(x_total_test)
        # 통계치 계산
        metrics = _calculate_metrics(y_test, test_pred)
        # 통계치 기준값 계산
        self._set_score_ref(ref=metrics)
        # 중요도 정렬
        importance = importance.sort_values(by=[importance_name], ascending=False)
        self.importance0 = importance
        # 전체 변수 설정
        self.total_var = importance.index.to_list()

    def _set_essential_var(self, essential_var: list, set_size=15):
        """Set Variable essential to represent the process or predict the target variable(Y)
           1. Set maximum number of variable for modelling
           2. Create a list of variables excluding essential variables from all variables """
        self.set_size = set_size
        if len(essential_var) >= 1:
            self.essential_var = essential_var
            model_inputs_init = copy.deepcopy(self.total_var)

            for ess in self.essential_var:
                model_inputs_init.remove(ess)
        else:
            self.essential_var = []
            model_inputs_init = copy.deepcopy(self.total_var)

        self.model_inputs_init = model_inputs_init
        # print(len(self.model_inputs_init))

    def forward_select(self, x_train, y_train, x_test, y_test):
        """Evaluate the performance of the model by adding variables one by one.
           0. k = 0
           1. Add the k th input variable among model_inputs list.
           2. Train the model and Calculate the performance of the model.
           3. Compare the performance of model with the reference value
           4. Decide whether to add or exclude variables"""
        # 모델의 입력 변수 설정(초기: [])
        model_inputs = self.model_inputs
        i = 0
        
        while i < len(self.total_var):  # 선택된 변수 크기 < 전체 변수
            selected = [var for var in self.selected]
            # i 번째 변수를 모델의 입력 변수에 추가
            selected.append(self.model_inputs_init[i])
            model_inputs = self.essential_var + selected
            train_x = x_train[model_inputs]
            test_x = x_test[model_inputs]

            # 모델링
            if self.model_type == "PLS":
                pls = PLS_TD()
                pls.fit(train_x, y_train)
                model = pls
            elif self.model_type == "LightGBM":
                lgbmr = LGBMR_TD()
                # lgbmr.find_param(x_data=train_x, y_data=y_train, trial_num=10)
                lgbmr.fit(x=train_x, y=y_train)
                # self.best_params = lgbmr.study.best_params
                # lgbmr.refit()
                model = lgbmr
            elif self.model_type == "XGBoost":
                xgbr = XGBR_TD()
                xgbr.find_param(x_data=train_x, y_data=y_train, trial_num=10)
                self.best_params = xgbr.study.best_params
                xgbr.refit()
                model = xgbr
            else:
                model = None
            # 예측값 계산
            test_pred = model.predict(test_x)
            # 통계치 계산
            metrics = _calculate_metrics(y_test, test_pred)
            # 모델 평가
            if len(selected) == 1:
                self.last_model_score = metrics
                self.selected = selected
            else:
                judge = _evaluate_model(metrics_test=metrics, metrics_ref=self.last_model_score)
                # 성능 향상
                if judge:
                    self.selected = selected
                    self.last_model_score = metrics
                else:
                    pass
            i += 1
            print(f"{i}/{len(self.model_inputs_init)}", f"선택된 변수의 개수: {len(model_inputs)}")

            if len(model_inputs) > self.set_size:
                # 최종 목표와 개발된 모델의 성능을 비교
                big_judge = _evaluate_model(metrics_test=metrics, metrics_ref=self.score_ref)
                #
                if big_judge:
                    return self.essential_var + self.selected
                else:
                    print("최종 모델의 성능에 비해 낮은 성능의 모델입니다.")
                    return self.essential_var + self.selected

    def refit(self, x_selected, y):
        if self.model_type == "PLS":
            pls = PLS_TD()
            pls.fit(x_selected, y)
            self.model = pls
        elif self.model_type == "LightGBM":
            # best_params = self.best_params
            lgbmr = LGBMR_TD()
            lgbmr.fit(x=x_selected, y=y, param=None)
            self.model = lgbmr
        elif self.model_type == "XGBoost":
            best_params = self.best_params
            xgbr = XGBR_TD()
            xgbr.fit(x_selected, y, param=best_params)
            self.model = xgbr

    def run(self, x_train, y_train,
            x_test, y_test, essentials=None, var_num=10):
        if essentials is None:
            essentials = []
        print("init")
        self._build_init_model(x_total_train=x_train, y_train=y_train,
                               x_total_test=x_test, y_test=y_test)
        print("set_essential")
        self._set_essential_var(essential_var=essentials, set_size=var_num)
        print("selecting....")
        self.selected_columns = self.forward_select(x_train, y_train, x_test, y_test)
        if self.selected_columns is None:
            print("선택된 변수가 없습니다. 전체 변수를 반환 합니다.")
            self.selected_columns = x_train.columns.to_list()
        x_train_selected = x_train[self.selected_columns]
        self.refit(x_train_selected, y_train)
