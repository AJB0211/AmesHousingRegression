import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression

from copy import deepcopy




from regressorBaseClass import customRegressor
#class MetaModel(BaseEstimator,RegressorMixin,TransformerMixin):
class MetaModel_lm(customRegressor):
    def __init__(self, in_df, models, sub_params, n_folds, qualPow, imputeDict):
        super(MetaModel_lm,self).__init__()
        self.qualPow = qualPow
        self.imputeDict = imputeDict
        # self.features = features
        self.models = models
        self.subparams = sub_params
        self.meta = None
        # self.model = self.meta # aliases shallow copied for use with base class
        self.n_folds = n_folds
        self.predBool = False

        from meta_features import impute_shell
        self._imputeVals = impute_shell(qualPow)
        tempDF = self._imputeVals(in_df)
        self.X = tempDF.drop(columns=["SalePrice"]).copy()
        self.y = np.log(tempDF.SalePrice).values.reshape(-1,1)
        self.pipeline_X = self._make_pipe()
        self.pipeline_y = RobustScaler()

    def _make_pipe(self):
        import meta_features as f
        nonePipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value="None"), OneHotEncoder(drop="first"))
        zeroPipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value=0), OneHotEncoder(drop="first", categories="auto"))
        scalePipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value=0), PowerTransformer())

        regressionPipeline = ColumnTransformer([
            ("setNone", nonePipeline, f.fillNone),
            # ("setZero", zeroPipeline, f.fillZeroCat),
            ("transformed", scalePipeline, f.fillZeroCont),
            # ("dictImputed", make_pipeline(self.dictImputer(f.imputeDict),
            #                               OneHotEncoder(drop="first")), list(f.imputeDict.keys())),
            # ("bool", "passthrough", f.imputeBool),
            ("categoricalInts", "passthrough", f.cat_to_int),
            # ("dropped", "drop", f.dropList)
        ], remainder="drop")
        return make_pipeline(regressionPipeline, RobustScaler())


    def genPreds(self,X,y):
        self.predBool = True
        self.model_list = [list() for i in self.models]
        folds = KFold(n_splits = self.n_folds, shuffle=True, random_state=6)

        oob_preds = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            for trainIdx, outIdx in folds.split(X):
                local_model = deepcopy(model)
                self.model_list[i].append(local_model)
                local_model.subset(trainIdx)
                local_model.fitModel(self.subparams[i])
                preds = local_model.predict(X.iloc[outIdx,:])
                oob_preds[outIdx,i] = preds.reshape(-1,)

        self.oob_preds = oob_preds

        # self.meta.fitModel(X, oob_preds, y)

    def fitModel(self,params):
        self._params = params
        self.meta = LinearRegression()

        if not self.predBool:
            self.genPreds(self.X,self.y)
        
        piped_X = self.pipeline_X.fit_transform(self.X)
        meta_X = np.column_stack([piped_X,self.oob_preds])
        piped_y = self.pipeline_y.fit_transform(self.y)

        self.meta.fit(meta_X,piped_y)



    def getTrainRsquared(self):
        piped_X = self.pipeline_X.transform(self.X)
        meta_X = np.column_stack([piped_X, self.oob_preds])
        piped_y = self.pipeline_y.transform(self.y)
        return self.meta.score(meta_X,piped_y)
        

    def predict(self,X):
        piped_X = self.pipeline_X.transform(self._imputeVals(X))
        pred_Data = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.model_list
        ])
        meta_X = np.column_stack([piped_X,pred_Data])
        preds = self.meta.predict(meta_X)
        return self._invert(preds)
