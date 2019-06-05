## Attempt to use simple average as an ensembling method

import pandas as pd
import numpy as np
from pathlib import Path
import re

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse
pd.set_option('mode.chained_assignment', None)

from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet,HuberRegressor,Ridge
from sklearn.ensemble import GradientBoostingRegressor

import lightgbm as lgb

import workup
import feature_engineering

trainData = pd.read_csv(Path("./data/train.csv"))
testData = pd.read_csv(Path("./data/test.csv"))

outliers = ((trainData.GrLivArea > 4000) & (trainData.SalePrice < 5E5))
trainData = trainData[~(outliers)]

imputeDict = workup.getImputeDicts(trainData)
qualPow = workup.getQualScale(trainData)


imputeVals = feature_engineering.impute_shell(qualPow=qualPow,**imputeDict)
pipeline_X = feature_engineering.make_pipe()
pipeline_y = RobustScaler()

print("Fitting pipelines")
train_X = pipeline_X.fit_transform(imputeVals(trainData.drop(columns=["SalePrice"])))
train_y = pipeline_y.fit_transform(np.log(trainData.SalePrice).values.reshape(-1,1)).reshape(-1,)

print("\n\nTransforming test data")
test_X = pipeline_X.transform(imputeVals(testData))


params_ElasticNet = {'alpha': 0.0037551020408163266,
                     'l1_ratio': 0.39166666666666666}
params_lgb = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 1000, 'min_child_samples': 10, 'min_child_weight': 0.001,
              'num_boost_round': 250, 'num_leaves': 5, 'reg_alpha': 0.05, 'reg_lambda': 5, 'subsample': 0.7, 'subsample_freq': 5}
params_ridge = {'alpha': 17.535070140280563}
params_gbr = {'learning_rate': 0.02894736842105263, 'loss': 'huber', 'max_depth': 3,
              'max_features': 'sqrt', 'min_samples_leaf': 10, 'n_estimators': 2000, 'subsample': 0.9}
params_svr = {'C': 4.684210526315789,
              'epsilon': 0.01, 'gamma': 'scale', 'kernel': 'rbf'}

ridge = Ridge(**params_ridge)
Enet = ElasticNet(**params_ElasticNet)
gbr = GradientBoostingRegressor(**params_gbr)
lgb = lgb.LGBMRegressor(objective="regression", metric="mse", boosting_type="dart",
                        device_type="cpu", tree_learner="feature", verbosity=-50, 
                        **params_lgb)
svr = SVR(**params_svr)

model_list = [ridge,Enet,gbr,lgb,svr]

for model in model_list:
    print(f'\nFitting model {model.__class__}')
    model.fit(train_X,train_y)

print("\nPrediction")
preds = np.column_stack([model.predict(test_X) for model in model_list]).mean(axis=1)
print(f'Shape of predictions: {preds.shape}')
preds = np.exp(pipeline_y.inverse_transform(preds.reshape(-1,1)))

submit_frame = pd.DataFrame()
submit_frame['Id'] = testData.Id
submit_frame['SalePrice'] = preds
submit_frame.to_csv('submission_ensemble_avg1.csv', index=False)



