import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

############# Must implement  ################################
# auxiliary features file which contains:
#      column selections
#      imputeShell which curries to imputeVals
# init which assigns:
#     pipeline_X
#     pipeline_y
#     _imputeVals
#     X  which is training X
#     y  which is training y
# gridSearch which assigns:
#     _gridSearchObject
#     _searchSpace
# fitModel which assigns
#     model
#     _params
################################################################


class customRegressor(object):
    def __init__(self):
        self._gridSearchObject = None
        self._searchSpace = None
        self._params = None
        self.model = None
        self.X = None
        self.y = None
        self.pipeline_y = None
        self.pipeline_X = None
        self._imputeVals = None

    # for ensembling
    def subset(self,Idx):
        print(self.__class__)
        self.X = self.X.reset_index().iloc[Idx,:]
        self.y = self.y[Idx,:]

    ## Transformer class for pipeline, imputing dict values on NAs
    class dictImputer(BaseEstimator, TransformerMixin):
        def __init__(self, dict_: dict):
            self.dict_ = dict_

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            for k, v in self.dict_.items():
                X[k] = X[k].fillna(v)
            return X[self.dict_.keys()]

    def predict(self, test_X):
        if self.model is not None:
            piped_X = self.pipeline_X.transform(self._imputeVals(test_X))
            preds = self.model.predict(piped_X)
            return self._invert(preds)
        else:
            raise ValueError("Must fit model first")

    # Root Mean Square Log Error
    def getRMSLE(self):
        if self.model is not None:
            piped_X = self.pipeline_X.transform(self.X)
            preds = self.pipeline_y.inverse_transform(self.model.predict(piped_X))
            return mean_squared_error(self.y, preds)
        else:
            raise ValueError("Must fit model first")

    def _invert(self, y):
        return np.exp(self.pipeline_y.inverse_transform(y))

    def gridSearch(self):
        pass
    
    def _make_pipe(self):
        pass
    
    def getBestParams(self):
        if self._gridSearchObject is not None:
            return self._gridSearchObject.best_params_
        else:
            raise ValueError("Must grid search first")

    def getBestScore(self):
        if self._gridSearchObject is not None:
            return self._gridSearchObject.best_score_
        else:
            raise ValueError("Must grid search first")

