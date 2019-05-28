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
from svr_pipeline import svReg


imputeDict = workup.getImputeDicts(trainData)
qualPow = workup.getQualScale(trainData)


svr = svReg(trainData,qualPow=qualPow,**imputeDict)

# search = {
#     "kernel": ["linear","poly","rbf"],
#     "gamma": ["auto"],
#     "C": np.linspace(0.2,50,10),
#     "epsilon": np.linspace(0.001,1.5,20)
#     }

search = {
    "kernel": ["rbf"],
    "gamma": ["auto"],
    "C": np.linspace(1,15,20),
    "epsilon": np.logspace(-5,-2,5)
    }

svr.gridSearch(search)
print("\n\n\n")

params = svr.getBestParams()
print(f'CV score: {svr.getBestScore()}')
print(params)



svr.fitModel(params)
print(f'Train R-squared: {svr.getTrainRsquared()}')
print(f'Train rmsle:     {svr.getRMSLE()}')

preds = svr.predict(testData)

submit_frame = pd.DataFrame()
submit_frame['Id'] = testData.Id
submit_frame['SalePrice'] = preds
submit_frame.to_csv('submission_svr1.csv', index=False)


#
# {'C': 5.733333333333333, 'epsilon': 0.001, 'gamma': 'auto', 'kernel': 'rbf'}
# {'C': 4.684210526315789, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'}
# 