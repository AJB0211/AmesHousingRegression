import pandas as pd
import numpy as np
from pathlib import Path
import re

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse


## These are local files
import workup
import feature_engineering

trainData = pd.read_csv(Path("./data/train.csv"))
testData = pd.read_csv(Path("./data/test.csv"))

outliers = ((trainData.GrLivArea > 4000) & (trainData.SalePrice < 5E5))
trainData = trainData[~(outliers)]

# This this retrieves a dictionary for mode imputation on MSZoning, Utilities, and LotFrontage features
imputeDict = workup.getImputeDicts(trainData)
# This returns the exponent to linearize the OverallQual feature
qualPow = workup.getQualScale(trainData)

## impute_shell:: (int,dict,dict,dict) => (pd.DataFrame => pd.DataFrame)
## imputeVals:: pd.DataFrame => pd.DataFrame 
## imputeVals is returned from a curried function impute_shell contained in feature_engineering.py
imputeVals = feature_engineering.impute_shell(qualPow=qualPow, **imputeDict)

## Pipeline structure is defined in feature_engineering
## Try to fork this file before making changes to track version control
pipeline_X = feature_engineering.make_pipe()
pipeline_y = make_pipeline(PowerTransformer(standardize=False),RobustScaler())

print("Fitting pipelines")
train_X = pipeline_X.fit_transform(imputeVals(trainData.drop(columns=["SalePrice"])))
train_y = pipeline_y.fit_transform(
    np.log(trainData.SalePrice.values.reshape(-1, 1))).reshape(-1,)

## Define invert function for retrieving predictions later
def invertPreds(y):
    return np.exp(pipeline_y.inverse_transform(y))

print("\n\nTransforming test data")
test_X = pipeline_X.transform(imputeVals(testData))


###################### GRID SEARCH #########################################
## Comment out this block if parameters are known
# search = {
#     "alpha": np.linspace(0.001, 0.01, 10),
#     "l1_ratio": np.linspace(0.2, 0.4, 10)
#     }

# ## n_jobs sets number of cores to parallelize on, set to 4 if you have 8 cores if you are getting bottlenecked
# ## verbose defines how much information is printed to console during search
# model_grid = GridSearchCV(ElasticNet(), search, scoring="neg_mean_squared_error", cv=5,n_jobs=-1, verbose=50)
# model_grid.fit(train_X,train_y)

# params = model_grid.best_params_

# print(f'Best grid search parameters:\n      {params}')
# print(f'Best grid search loss:\n         {model_grid.best_score_}')


###########################################################################
## Paste parameter results and scores here for posterity
params = {'alpha': 0.001, 'l1_ratio': 0.4} 
## Score: -0.0541876633514389

model = ElasticNet(**params)
model.fit(train_X,train_y)

train_preds = model.predict(train_X)

# Mean Squared Log Error
print(f'Train MSLE:  {mse(train_y,train_preds)}')


preds = invertPreds(model.predict(test_X).reshape(-1,1))

## This is formatting for submission to Kaggle
submit_frame = pd.DataFrame()
submit_frame['Id'] = testData.Id
submit_frame['SalePrice'] = preds
submit_frame.to_csv('example submission.csv', index=False)


