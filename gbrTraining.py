from gbr_pipeline import GradBoostReg
import workup
from pathlib import Path
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


trainData = pd.read_csv(Path("./data/train.csv"))
testData = pd.read_csv(Path("./data/test.csv"))


imputeDict = workup.getImputeDicts(trainData)
qualPow = workup.getQualScale(trainData)


lm = GradBoostReg(trainData, qualPow=qualPow, **imputeDict)

# search = {
#     "loss": ["huber"],
#     "learning_rate": np.logspace(-3,0,10),
#     "n_estimators": [100,500,100],
#     "subsample": [0.5,0.7,0.9],
#     "max_depth": [3,5,7],
#     "max_features": ["sqrt"],
#     "min_samples_leaf": [20]
# }

# # Train R-squared: 0.9526557004763557
# # Train rmsle:     0.0075491672112742905

# params = {'learning_rate': 0.046415888336127774, 'loss': 'huber', 'max_depth': 5,
#           'max_features': 'sqrt', 'min_samples_leaf': 20, 'n_estimators': 500, 'subsample': 0.9}

# search = {
#     "loss": ["huber"],
#     "learning_rate": np.linspace(0.01,0.1,20),
#     "n_estimators": [100, 500, 1000],
#     "subsample": [0.9,1.0],
#     "max_depth": [5],
#     "max_features": ["sqrt"],
#     "min_samples_leaf": [10,20,30]
# }

# # Train R-squared: 0.9655910656109367
# # Train rmsle:     0.005486590822514529

# search = {'learning_rate': [0.02894736842105263], 
#             'loss': ['huber'], 
#             'max_depth': [3,4,5],
#           'max_features': ['sqrt'], 
#           'min_samples_leaf': [10,20], 
#           'n_estimators': [1500,2000,2500], 
#           'subsample': [0.9]}

# # Train R-squared: 0.9739725816143878
# # Train rmsle:     0.004150137090372493

# lm.gridSearch(search)
# print("\n\n\n")

# params = lm.getBestParams()
# print(f'CV score: {lm.getBestScore()}')
# print(params)

params = {'learning_rate': 0.02894736842105263, 'loss': 'huber', 'max_depth': 5,
          'max_features': 'sqrt', 'min_samples_leaf': 10, 'n_estimators': 1500, 'subsample': 0.9}

params = {'learning_rate': 0.02894736842105263, 'loss': 'huber', 'max_depth': 3,
          'max_features': 'sqrt', 'min_samples_leaf': 10, 'n_estimators': 2000, 'subsample': 0.9}


lm.fitModel(params)
print(f'Train R-squared: {lm.getTrainRsquared()}')
print(f'Train rmsle:     {lm.getRMSLE()}')

preds = lm.predict(testData)

submit_frame = pd.DataFrame()
submit_frame['Id'] = testData.Id
submit_frame['SalePrice'] = preds
submit_frame.to_csv('submission_gbr2.csv', index=False)
