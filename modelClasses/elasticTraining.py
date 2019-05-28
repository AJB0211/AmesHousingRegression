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
from ElasticNet_pipeline import ElasticReg


imputeDict = workup.getImputeDicts(trainData)
qualPow = workup.getQualScale(trainData)


lm = ElasticReg(trainData,qualPow=qualPow,**imputeDict)

# search = {
#     "alpha": np.logspace(-5, 4, 25), 
#     "l1_ratio": np.linspace(0, 1, 25)}

# search = {
#     "alpha": np.logspace(-4, 1, 25),
#     "l1_ratio": np.linspace(0, 0.5, 25)}

# search = {
#     "alpha": np.linspace(0.001, 0.01, 50),
#     "l1_ratio": np.linspace(0.2, 0.4, 25)}

# lm.gridSearch(search)
# print("\n\n\n")

# params = lm.getBestParams()
# print(f'CV score: {lm.getBestScore()}')
# print(params)


# params = {'alpha': 0.0005623413251903491, 'l1_ratio': 1.0}
params = {'alpha': 0.0037551020408163266, 'l1_ratio': 0.39166666666666666}

lm.fitModel(params)
print(f'Train R-squared: {lm.getTrainRsquared()}')
print(f'Train rmsle:     {lm.getRMSLE()}')

preds = lm.predict(testData)

submit_frame = pd.DataFrame()
submit_frame['Id'] = testData.Id
submit_frame['SalePrice'] = preds
submit_frame.to_csv('submission_elastic1.csv', index=False)


