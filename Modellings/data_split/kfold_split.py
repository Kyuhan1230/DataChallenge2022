import numpy as np
from sklearn.model_selection import KFold


class KfoldSplit:
    def __init__(self, n_splits=None):
        self.train_cv_indices_list = []
        self.test_cv_indices_list = []
        self.data_size = None
        if n_splits is None:
            n_splits = 5    # Number of train/cv/test folds
        self.n_splits = n_splits

    def set_n_splits(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self):
        return self.n_splits

    def split_data(self, data):
        self.data_size = len(data)
        kfold = KFold(n_splits=self.n_splits)
        train_test_split = kfold.split(data)
        result = {}
        i = 0
        data = np.array(data)

        for train_cv_indices, test_cv_indices in train_test_split:
            # First, we split Train + CV and Test
            train_cv = data[train_cv_indices, :]
            test_cv = data[test_cv_indices, :]
            result[i] = [train_cv, test_cv]

            self.train_cv_indices_list.append(train_cv_indices)
            self.test_cv_indices_list.append(test_cv_indices)
            i += 1
        return result
