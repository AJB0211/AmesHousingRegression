import numpy as np 
import pandas as pd

#import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# Disable warnings
pd.set_option('mode.chained_assignment', None)

from pathlib import Path


trainData = pd.read_csv(Path("./data/train.csv"))
testData = pd.read_csv(Path("./data/test.csv"))


import workup
from ridge_pipeline import RidgeReg


imputeDict = workup.getImputeDicts(trainData)
qualPow = workup.getQualScale(trainData)


ridge_model = RidgeReg(trainData,qualPow=qualPow,**imputeDict)


search = {
"alpha": np.linspace(0,50,500),
}

# ridge_model.gridSearch(search)
# print("\n\n\n")

# params_Ridge = ridge_model.getBestParams()
# print(f'CV score: {ridge_model.getBestScore()}')
# print(params_Ridge)

params_Ridge = {'alpha': 17.535070140280563}
ridge_model.fitModel(params_Ridge)
print(f'LM R-squared: {ridge_model.getTrainRsquared()}')
print(f'LM RMSLE:     {ridge_model.getRMSLE()}')

preds = ridge_model.predict(testData)

submit_frame = pd.DataFrame()
submit_frame['Id'] = testData.Id
submit_frame['SalePrice'] = preds
submit_frame.to_csv('submission_ridge1.csv', index=False)
