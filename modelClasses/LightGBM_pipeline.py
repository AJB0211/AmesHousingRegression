import pandas as pd
import numpy as np
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse
pd.set_option('mode.chained_assignment', None)

import lightgbm as lgb

# qualPow in constructor does nothing, it is for use in ensembling
from regressorBaseClass import customRegressor
class lgbmReg(customRegressor):
    def __init__(self, in_df, qualPow, zoning, utilities, frontage):
        super(lgbmReg,self).__init__()

        from lgbm_features import imputeShell
        ## This is a curried function call
        self._imputeVals = imputeShell(zoning=zoning,utilities=utilities,frontage=frontage)
        tempDF = self._imputeVals(in_df.copy())

        self.X = tempDF.drop(columns=["SalePrice"]).copy()
        self.y = np.log(tempDF.SalePrice.values.reshape(-1, 1))

        self.pipeline_X = self._make_pipe()
        self.pipeline_X.fit(self.X)
        self.pipeline_y = StandardScaler()
        self.pipeline_y.fit(self.y)

    def _make_pipe(self):
        import lgbm_features as ft
        nonePipeline = make_pipeline(SimpleImputer(
        strategy="constant", fill_value="None"), OneHotEncoder(drop="first"))
        zeroPipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value=0), OneHotEncoder(drop="first", categories="auto"))
        scalePipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value=0), PowerTransformer())#, StandardScaler())

        regressionPipeline = ColumnTransformer([
            ("setNone", nonePipeline, ft.fillNone),
            ("setZero", zeroPipeline, ft.fillZeroCat),
            ("transformed", scalePipeline, ft.fillZeroCont),
            ("dictImputed", make_pipeline(self.dictImputer(ft.imputeDict),
                                            OneHotEncoder(drop="first")), list(ft.imputeDict.keys())),
            ("bool", "passthrough", ft.imputeBool),
            ("categoricalInts", "passthrough", ft.cat_to_int),
            # ("selected","passthrough",selected),
            ("dropped", "drop", ft.dropList)
        ], remainder="drop")
        #return regressionPipeline
        return make_pipeline(regressionPipeline, RobustScaler())



    def gridSearch(self,params,cv=3,njobs=-1,verbose=50,device_type="cpu",boosting_type="gbdt",verbosity=1):
        gridRegressor = lgb.LGBMRegressor(
            objective="regression", metric="mse", boosting_type=boosting_type, device_type=device_type, tree_learner="feature", verbosity=verbosity)

        piped_X = self.pipeline_X.transform(self.X)
        piped_y = self.pipeline_y.transform(self.y)

        self._searchSpace = params
        self._gridSearchObject = GridSearchCV(gridRegressor,params,cv=cv)
        self._gridSearchObject.fit(piped_X,piped_y)
    

    def fitModel(self,params, boosting_type = "dart", device_type = "cpu",verbosity = -5):
        self._params = params
        self.model = lgb.LGBMRegressor(objective="regression", metric="mse", boosting_type=boosting_type,
                                 device_type=device_type, tree_learner="feature", verbosity=verbosity, **params)

        piped_X = self.pipeline_X.transform(self.X)
        piped_y = self.pipeline_y.transform(self.y).reshape(-1,)

        self.model.fit(piped_X, piped_y)
