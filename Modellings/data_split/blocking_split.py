import numpy as np
from sklearn.model_selection import TimeSeriesSplit


class BlockingTimeSeriesSplit:
    def __init__(self, n_splits=5):
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
    
    def _get_test_size(self, data):
        train_test_split = TimeSeriesSplit(self.n_splits+1).split(data)
        for train_cv_indices, test_cv_indices in train_test_split:
            test_size = len(test_cv_indices)
            break
        return test_size
    
    def split_data(self, data):
        self.data_size = len(data)
        test_size = self._get_test_size(data)

        stops = [len(data) - i * test_size for i in range(self.n_splits)]
        stops.reverse()
        
        train_size = stops[0] - test_size
        starts = [stop - train_size - test_size for stop in stops]
        mids = [stop - test_size for stop in stops]
        
        result = {}
        indices = np.arange(len(data))
        i = 0
        data = np.array(data)

        for start, mid, stop in zip(starts, mids, stops):
            train_cv_indices, test_cv_indices = indices[start: mid], indices[mid: stop]
            
            train_cv = data[train_cv_indices, :]
            test_cv = data[test_cv_indices, :]
            result[i] = [train_cv, test_cv]
            
            self.train_cv_indices_list.append(train_cv_indices)
            self.test_cv_indices_list.append(test_cv_indices)
            i += 1
        return result
