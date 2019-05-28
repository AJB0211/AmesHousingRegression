import pandas as pd
import numpy as np


from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error


## Class for bundling linear regression operations

from regressorBaseClass import customRegressor
class ElasticReg(customRegressor):
    def __init__(self, in_df,zoning,utilities,frontage,qualPow):

        super(ElasticReg,self).__init__()
        from lm_features import impute_shell
        ## Because we're currying in python now
        self._imputeVals = impute_shell(frontage = frontage,zoning=zoning,utilities=utilities,qualPow=qualPow)
        tempDF = self._imputeVals(in_df.copy())
        self.X = tempDF.drop(columns=["SalePrice"]).copy()
        self.y = np.log(tempDF.SalePrice.values.reshape(-1, 1))


        self.pipeline_X = self._make_pipe()
        self.pipeline_y = StandardScaler()


    def _rmOutliers(self,x ,y):
        outliers = ((y > 4000) & (y < 5E5))
        out = x[~(outliers)]

        return out

    def _make_pipe(self):
        import lm_features as f
        nonePipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value="None"), OneHotEncoder(drop="first"))
        zeroPipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value=0), OneHotEncoder(drop="first", categories="auto"))
        scalePipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value=0), PowerTransformer())

        regressionPipeline = ColumnTransformer([
            ("setNone", nonePipeline, f.fillNone),
            ("setZero", zeroPipeline, f.fillZeroCat),
            ("transformed", scalePipeline, f.fillZeroCont),
            ("dictImputed", make_pipeline(self.dictImputer(f.imputeDict),
                                          OneHotEncoder(drop="first")), list(f.imputeDict.keys())),
            ("bool", "passthrough", f.imputeBool),
            ("categoricalInts", "passthrough", f.cat_to_int),
            ("dropped", "drop", f.dropList)
        ], remainder="drop")
        return make_pipeline(regressionPipeline,RobustScaler())

    def gridSearch(self, params, cv=5, njobs=-1, verbose=50):
        self._searchSpace = params

        piped_X = self._rmOutliers(self.X,self.y)
        piped_X = self.pipeline_X.fit_transform(piped_X)
        piped_y = self.pipeline_y.fit_transform(self.y)

        self._gridSearchObject = GridSearchCV(
            ElasticNet(), params, cv=cv, scoring="neg_mean_squared_error", n_jobs=njobs, verbose=verbose)
        self._gridSearchObject.fit(piped_X, piped_y)



    def fitModel(self, params):
        self.model = ElasticNet()
        self._params = params

        piped_X = self._rmOutliers(self.X,self.y)
        piped_X = self.pipeline_X.fit_transform(piped_X)
        piped_y = self.pipeline_y.fit_transform(self.y)

        self.model.set_params(**params)
        self.model.fit(piped_X, piped_y)


    def getTrainRsquared(self):
        piped_X = self._rmOutliers(self.X,self.y)
        piped_X = self.pipeline_X.transform(piped_X)
        piped_y = self.pipeline_y.transform(self.y)
        return self.model.score(piped_X, piped_y)


