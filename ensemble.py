from pathlib import Path
import numpy as np 
import pandas as pd

#import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
# Disable warnings
pd.set_option('mode.chained_assignment', None)

trainData = pd.read_csv(Path("./data/train.csv"))
testData = pd.read_csv(Path("./data/test.csv"))


import workup
from ElasticNet_pipeline import ElasticReg
from LightGBM_pipeline import lgbmReg
from ridge_pipeline import RidgeReg
from gbr_pipeline import GradBoostReg
from svr_pipeline import svReg
from metaModel import MetaModel_Lasso


imputeDict = workup.getImputeDicts(trainData)
qualPow = workup.getQualScale(trainData)


params_ElasticNet = {'alpha': 0.0037551020408163266,
                     'l1_ratio': 0.39166666666666666}
params_lgb = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 1000, 'min_child_samples': 10, 'min_child_weight': 0.001,
              'num_boost_round': 250, 'num_leaves': 5, 'reg_alpha': 0.05, 'reg_lambda': 5, 'subsample': 0.7, 'subsample_freq': 5}
params_ridge = {'alpha': 17.535070140280563}
params_gbr = {'learning_rate': 0.02894736842105263, 'loss': 'huber', 'max_depth': 3,
              'max_features': 'sqrt', 'min_samples_leaf': 10, 'n_estimators': 2000, 'subsample': 0.9}
params_svr = {}

paramList = [params_ElasticNet,
            params_lgb,
            params_ridge,
            params_gbr#, params_svr
            ]


lm = ElasticReg(trainData,qualPow=qualPow,**imputeDict)
lgb = lgbmReg(trainData,**imputeDict)
ridge = RidgeReg(trainData,qualPow=qualPow,**imputeDict)
gbr = GradBoostReg(trainData, qualPow=qualPow, **imputeDict)
svr = svReg(trainData,qualPow=qualPow, **imputeDict)

models = [lm,lgb,ridge,gbr]#,svr]

# maybe make n_folds tunable????
meta = MetaModel_Lasso(trainData,models, n_folds = 5, qualPow=qualPow, imputeDict=imputeDict)





lm.fitModel(params_ElasticNet)
print(f'LM R-squared: {lm.getTrainRsquared()}')
print(f'LM RMSLE:     {lm.getRMSLE()}')


lgb.fitModel(params_lgb)


print(f'LGBM RMSLE: {lgb.getRMSLE()}')
