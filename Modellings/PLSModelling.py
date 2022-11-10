"""
2. PLS Modeling
    - _determine_comp
    - fit
    - predict
    - calc_metrics
    - get_coefs
    - get_vips
    - get_PLS_stats
    - get_contribution
"""
import math
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List
from scipy.stats import chi2
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression

from Modellings.metrics import ModelScore


class PLS_TD:
    def __init__(self):
        self.pls_model = None
        self.y_train = None
        self.x_train = None
        self.n_comp = None

    def determine_comp(self, x_train, y_train):
        """determine the number of latent variable : AutoFit of SIMCA
            object: Find the number with R2 Differences Less than 1%
            R2: Coef Of Determination"""
        x_num = x_train.shape[1]
        r_list = []
        if x_num > 1:
            for i in range(x_num):
                pls = PLSRegression(n_components=i + 1)
                pls.fit(x_train, y_train)
                y_train_pred = pls.predict(x_train)
                r_list.append(r2_score(y_train, y_train_pred))

            diff_r = []
            for i in range(len(r_list) - 1):
                diff_r.append(r_list[i + 1] - r_list[i])
            diff_r.insert(0, r_list[0] - 0)

            for i in range(len(r_list) - 1):
                if diff_r[i] > 0.01:
                    n_comp = x_num  # 만일 0.01보다 차이가 안나는 경우가 생긴다면. n_comp=x_num으로 설정하자.
                else:
                    n_comp = i
                    break
        else:
            n_comp = x_num

        self.n_comp = n_comp

        self.x_train, self.y_train = x_train, y_train

        print("Suggested number of PLS Component:", self.n_comp)

    def fit(self, x, y, n_comp=None):
        """fit the pls model"""
        if n_comp is None:
            self.determine_comp(x, y)
            n_comp = self.n_comp

        if type(x) == pd.core.frame.DataFrame:
            setattr(self, 'x_columns', x.columns.to_list())
        else:
            setattr(self, 'x_columns', [f'X{col + 1}' for col in range(x.shape[1])])

        self.pls_model = PLSRegression(n_components=n_comp)
        self.pls_model.fit(x, y)

    def _scale_x(self, stats: dict, x: np.ndarray) -> np.ndarray:
        x_mean, x_std = stats['x_mean'], stats['x_std']
        x_norm = (x - x_mean) / x_std
        x_norm  = np.array(x_norm)
        return x_norm

    def _unscaled_x(self, stats: dict, x: np.ndarray) -> np.ndarray:
        x_mean, x_std = stats['x_mean'], stats['x_std']
        x_denorm = x * x_std + x_mean
        return x_denorm

    def predict(self, x_test):
        """predict value using trained model"""
        y_pred = self.pls_model.predict(x_test)
        return y_pred

    def calculate_coef(self):
        """calculate pls coefficients in unscaled form"""
        avg_list = self.pls_model.x_mean_  # x평균
        std_list = self.pls_model.x_std_  # x표준편차
        coef_list = self.pls_model.coef_  # scaled Coefficient

        for i in range(len(std_list)):
            if float(std_list[i]) == 0:
                std_list[i] = 0.000000000000001

        de_coef: list = coef_list.T / std_list
        sum_decoef = np.sum(coef_list.T * avg_list / std_list, axis=1)
        denorm_c = self.pls_model.y_mean_ - sum_decoef

        coef = list(de_coef[0])
        coef.append(denorm_c[0])
        return coef

    def get_coef(self, scale_option: str = 'Unscale'):
        """convert coef to dataframe from list using x_columns"""
        if scale_option == 'Unscale':
            coef = self.calculate_coef()
            coef_columns = self.x_columns + ['Const']
        else:
            coef = [float(c) for c in self.pls_model.coef_]
            coef_columns = self.x_columns

        coef_ = {coef_columns[i]: coef[i] for i in range(len(coef))}
        coef_ = pd.DataFrame.from_dict([coef_])
        return coef_

    def calc_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """calculate metrics: R2Y, RA, MSE, Y_Avg, Y_Std, |Error|avg, Error std """
        ms = ModelScore()
        metrics = ms.calc_y_metrics(y_true=y_true, y_pred=y_pred)
        return metrics

    def calculate_vip(self) -> pd.DataFrame:
        """calculate vip"""
        t = self.pls_model.x_scores_
        w = self.pls_model.x_weights_
        q = self.pls_model.y_loadings_
        p, h = w.shape
        vips = np.zeros((p,))
        s = np.diag(np.dot(np.dot(np.dot(t.T, t), q.T), q)).reshape(h, -1)
        total_s = np.sum(s)

        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            vips[i] = np.sqrt(p * (np.dot(s.T, weight)) / total_s)

        vips = list(vips)
        col = self.x_columns
        tmp = vips.copy()
        tmp.sort(reverse=True)
        re_col = []

        for i in range(len(col)):
            col_idx = vips.index(tmp[i])
            re_col.append(col[col_idx])
        vip = pd.DataFrame(vips, index=self.x_columns, columns=['VIP value'])

        return vip

    def get_PLS_stats(self) -> dict:
        x_mean = self.pls_model.x_mean_
        x_std = self.pls_model.x_std_

        y_mean = self.pls_model.y_mean_
        y_std = self.pls_model.y_std_

        loading_p = self.pls_model.x_loadings_
        loading_q = self.pls_model.y_loadings_
        score_t = self.pls_model.x_scores_
        weight_w = self.pls_model.x_weights_
        weight_star = self.pls_model.x_rotations_

        lv_num = self.n_comp
        x_num = int(x_mean.shape[0])

        stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
                 "loading_p": loading_p, "loading_q": loading_q, "score_t": score_t,
                 "weight_w": weight_w, "weight_star": weight_star,
                 "lv_num": lv_num, "x_num": x_num}

        return stats

    def get_x_predict(self, x=None):
        stats = self.get_PLS_stats()
        w_star = stats['weight_star']
        loading_p = stats['loading_p']
        if x is not None:
            x_norm = self._scale_x(stats=stats, x=x)
            score_t = self._get_t_predict(w_star=w_star, x=x_norm)
        else:
            score_t = self.get_PLS_stats()['score_t']
        x_hat = np.matmul(score_t, loading_p.T)
        x_hat = self._unscaled_x(stats=stats, x=x_hat)
        return x_hat

    def _get_t_predict(self, w_star, x):
        t_hat = np.matmul(x, w_star)
        return t_hat

    def get_hotellingT2(self, x=None) -> np.ndarray:
        t_std = self._get_t_std()
        lv_num = self.n_comp
        stats = self.get_PLS_stats()
        w_star = stats['weight_star']

        if x is not None:
            x_norm = self._scale_x(stats=stats, x=x)
            t_score = self._get_t_predict(w_star=w_star, x=x_norm)
        else:
            t_score = self.get_PLS_stats()['score_t']

        T2_df = (t_score / t_std) * (t_score / t_std)
        T2 = 0
        for i in range(lv_num):
            T2 += T2_df[:, i]

        return T2

    def get_SPE(self, x=None) -> np.ndarray:
        pls_stats = self.get_PLS_stats()

        if x is not None:
            x_hat = self.get_x_predict(x)
            x_hat = self._scale_x(stats=pls_stats, x=x_hat)
        else:
            loading_p = pls_stats['loading_p']
            score_t = pls_stats['score_t']
            x = self._scale_x(stats=pls_stats, x=self.x_train)
            x_hat = np.matmul(score_t, loading_p.T)

        x_error = x - x_hat
        sum_x_error = x_error * x_error

        spe = 0
        x_num = pls_stats['x_num']
        for i in range(x_num):
            spe += sum_x_error[:, i]
        return spe

    def _get_selected(self, index_g1: list, x_data: pd.DataFrame, index_g2=None) -> List[np.ndarray]:
        x_index = x_data.index.to_list()
        g1_index = [x_index[g] for g in index_g1]
        x_g1 = x_data.loc[g1_index]

        if index_g2 is not None:
            g2_index = [x_index[g] for g in index_g2]
            x_g2 = x_data.loc[g2_index]
        else:
            x_g2 = x_data

        x_g1 = np.array(x_g1)
        x_g2 = np.array(x_g2)
        return [x_g1, x_g2]

    def _get_average_grouped(self, selecteds: List[np.ndarray]) -> List[np.ndarray]:
        x_g1, x_g2 = selecteds[0], selecteds[1]
        g1_mean, g2_mean = x_g1.mean(axis=0), x_g2.mean(axis=0)
        return [g1_mean, g2_mean]

    def _calc_contribution(self, x_num: int, g1_t: np.ndarray, g2_t: np.ndarray,
                           g1_mean_norm: np.ndarray, g2_mean_norm: np.ndarray,
                           t_std: list, lv_num: int,w_star: np.ndarray,
                           calc_type: str='Weight 1') -> list:
        cts = []
        print(calc_type)
        if calc_type == "T2 Contribution":
            for j in range(x_num):
                res1, res2, = 0, 0
                for k in range(lv_num):
                    term1 = g1_t[k] / (t_std[k] * t_std[k])
                    term2 = np.dot(g1_mean_norm[j], w_star[j, k])

                    term3 = g2_t[k] / (t_std[k] * t_std[k])
                    term4 = np.dot(g2_mean_norm[j], w_star[j, k])

                    a = term1 * term2
                    b = term3 * term4

                    res1 += a
                    res2 += b

                ct = res2 - res1
                cts.append(float(ct))
            return cts
            
        elif calc_type == "Weight 1":
            # ct = (Xj1_normalized - Xj2_normalized) * Weight * Std(t1) * Std(t1)
            for j in range(x_num):
                ct = (g1_mean_norm[j] - g2_mean_norm[j]) * abs(w_star.T[j, 0]) * t_std[0] * t_std[0]
                cts.append(float(ct))
            return cts
                           
        elif calc_type == "Normalized":
           # ct = (Xj1_normalized - Xj2_normalized)
            for j in range(x_num):
                ct = (g1_mean_norm[j] - g2_mean_norm[j])
                cts.append(float(ct))
            return cts
                           
        else:
            raise TypeError("The Type should be Weight 1, Normalized, T2 Contribution.")

    def get_contribution(self, index: List[list], x_data) -> list:
        pls_stats = self.get_PLS_stats()
        x_num = pls_stats['x_mean'].shape[0]
        w_star = pls_stats['weight_star']
        t_std = self._get_t_std()
        lv_num = len(t_std)

        index_g1 = index[0]

        if len(index)==1:
            index_g2 = None
        else:
            index_g2 = index[1]

        selecteds = self._get_selected(index_g1=index_g1, index_g2=index_g2, x_data=x_data)
        [g1_mean, g2_mean] = self._get_average_grouped(selecteds)
        g1_mean_norm = self._scale_x(stats=pls_stats, x=g1_mean)
        g2_mean_norm = self._scale_x(stats=pls_stats, x=g2_mean)

        g1_t = np.dot(g1_mean_norm, w_star)
        g2_t = np.dot(g2_mean_norm, w_star)

        cts = self._calc_contribution(x_num=x_num, g1_t=g1_t, g2_t=g2_t,
                                      g1_mean_norm=g1_mean_norm,g2_mean_norm=g2_mean_norm,
                                      t_std=t_std, lv_num=lv_num, w_star=w_star)
        return cts

    def _get_t_std(self):
        pls_stats = self.get_PLS_stats()
        t = pls_stats['score_t']
        t_std = []

        for i in range(len(t[0, :])):
            std = t[:, i].std()
            t_std.append(std)

        return t_std

    def get_score_oval_border(self) -> List[np.ndarray]:
        data_size = self.get_PLS_stats()['score_t'].shape[0]
        t1_std, t2_std = self._get_t_std()[0], self._get_t_std()[1]

        dfn = 2
        dfd = data_size - 2

        F = scipy.stats.f.ppf(0.95, dfn, dfd)
        FF = 2 * ((data_size - 1) / (data_size - 2))

        A = math.sqrt(t1_std * t1_std * F * FF)
        B = math.sqrt(t2_std * t2_std * F * FF)
        angle = np.linspace(0, 2 * np.pi, 100)

        oval_border_x = A * np.cos(angle)
        oval_border_y = B * np.sin(angle)
        return [oval_border_x, oval_border_y]

    def get_T2_limit(self, lim_95=None, lim_99=None):
        lv_num = self.get_PLS_stats()['lv_num']
        if lim_95 is None:
            lim_95 = scipy.stats.chi2.ppf(0.95, lv_num)
        if lim_99 is None:
            lim_99 = scipy.stats.chi2.ppf(0.99, lv_num)
        result = [lim_95, lim_99]
        return result

    def get_SPE_limit(self, lim_95=None, lim_99=None):
        spe_train = self.get_SPE()
        spe_train_std = spe_train.std()
        spe_train_mean = spe_train.mean()

        a = (spe_train_std * spe_train_std) / (2 * spe_train_mean)
        A = (2 * spe_train_mean * spe_train_mean) / (spe_train_std * spe_train_std)

        if A <= 1:
            A = 1

        spe_lim_95 = a * chi2.ppf(0.95, A)
        spe_lim_99 = a * chi2.ppf(0.99, A)

        if lim_95 is None:
            lim_95 = spe_lim_95
        if lim_99 is None:
            lim_99 = spe_lim_99

        result = [lim_95, lim_99]

        return result