# from statsmodels.api import OLS
# import statsmodels.api as sm
import numpy as np
import pandas as pd

#import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as mse

import lightgbm as lgb

from pathlib import Path
import re

# Disable warnings
pd.set_option('mode.chained_assignment', None)

dataDir = Path("./data")
#trainData = pd.read_csv("./data/train.csv")
#testData = pd.read_csv("./data/test.csv")

trainData = pd.read_csv(Path("./data/train.csv"))
testData = pd.read_csv(Path("./data/test.csv"))

trainData.columns[trainData.columns.str.contains("SF")]

s = "MSZoning, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, SaleType, SaleCondition"

categories = re.split(",\s?",s)

toInt = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
         'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

lgb_cats = [i for i in categories if i not in toInt] + ["MSSubClass", "MoSold"]

#outliers = ((trainData.GrLivArea > 4000) & (trainData.SalePrice < 5E5))
#trainData = trainData[~(outliers)]

zoning = trainData.groupby("Neighborhood").MSZoning.apply(
    lambda x: x.value_counts().sort_values().index[0]).to_dict()
utilities = trainData.groupby("Neighborhood").Utilities.apply(
    lambda x: x.value_counts().sort_values().index[0]).to_dict()
frontage = trainData.groupby("Neighborhood").LotFrontage.apply(
    lambda x: x.value_counts().sort_values().index[0]).to_dict()

## Function for values that are manually imputed
def imputeVals(in_df):
    df = in_df.copy()
    df.LotFrontage = df.LotFrontage.fillna(df.Neighborhood.map(frontage))
    df.MSZoning = df.MSZoning.fillna(df.Neighborhood.map(zoning))
    df.Utilities = df.Utilities.fillna(df.Neighborhood.map(utilities))
    df["CentralAir"] = (df["CentralAir"] == "Y")
    df.MSSubClass[df.MSSubClass == 150] = 120
    df["sinMonth"] = df.MoSold.apply(lambda x: np.sin(np.pi*x/12))
    #df["scaledOverallQual"] = df.OverallQual.apply(lambda x: x**qualPow)
    df.Condition1[df.Condition1 == "RRNe"] = "RRNn"
    df.OverallCond[df.OverallCond < 3] = 3
    df.Exterior1st[df.Exterior1st == "AsphShn"] = "AsbShng"
    df.Exterior1st[df.Exterior1st == "ImStucc"] = "Stone"
    df.Heating[df.Heating == "Floor"] = "Grav"
    df.Heating[df.Heating == "OthW"] = "Wall"
    #df.Electrical[df.Electrical != "SBrkr"] = "Oth"
    df["HasPool"] = df.PoolQC.notnull()
    df["garageDiff"] = df.GarageYrBlt - df.YearBuilt
    df["remodDiff"] = df.YearRemodAdd - df.YearBuilt

    toInt = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
             'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    intDict = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    for i in toInt:
        df[i] = df[i].apply(lambda x: intDict.get(x, 0))

    df["SF"] = df.TotalBsmtSF + df["1stFlrSF"] + df["2ndFlrSF"]
    df["numBaths"] = df.BsmtFullBath + 0.5*df.BsmtHalfBath + df.FullBath + 0.5*df.HalfBath
    df["2story"] = (df["2ndFlrSF"] != 0)
    df["hasBsmt"] = (df["TotalBsmtSF"] != 0)

    return df

####################################################################################################
############### THIS IS WHERE YOU SELECT WHICH FEATURES ARE INCLUDED IN THE MODEL ##################
selected = ["sinMonth" ]
BsmtDropped = ["BsmtFinSF1", "BsmtUnfSF", "BsmtFinSF2"]
# values that null is filled with "None" then get one-hot encoded
fillNone = ["MasVnrType", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageType", "GarageFinish",
            "Fence", "MiscFeature", "MasVnrType", "LotShape", "LandSlope", "Neighborhood",
            "Condition1", "LotConfig", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
            "Foundation", "Heating", "SaleCondition", "Electrical", "PavedDrive", "Alley",
            "Utilities", "Condition2"]

# Categorical variables represented as integers
cat_to_int = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
              'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

# ordinal categorical variables
fillZeroCat = ["MSSubClass"]

 
# continuous variables with missing values that are zero
diffVars = ["garageDiff", "remodDiff"]
rmVars1 = ["MiscVal", "LowQualFinSF","BsmtFullBath", "HalfBath","BsmtHalfBath"]
fillZeroCont = ["MasVnrArea", "GarageArea", "GrLivArea", "1stFlrSF", "2ndFlrSF", "LotFrontage", 
                "TotalBsmtSF", "WoodDeckSF", "OpenPorchSF", "PoolArea", "BedroomAbvGr", "YrSold", "MoSold", 
                "LotArea", "EnclosedPorch", "OverallCond", "OverallQual", 
                "LotFrontage", "FullBath", "Fireplaces", "TotRmsAbvGrd", "TotalBsmtSF", "numBaths", "Fireplaces", "TotRmsAbvGrd"]   + rmVars1 

# variables that need differences between reference engineered
imputeDiff = [("GarageYrBlt", "YearBuilt"), ("YearRemodAdd", "YearBuilt")]

# imputed to boolean: passthrough
imputeBool = ["CentralAir", "HasPool", "2story", "hasBsmt"]

# categories that we need to know if they were imputed
imputeUnknown = []

# List of values taken out to be onehotencoded with a list argument
# Due to missing values in test data
handleMissingInt = ["GarageCars" ]
handleMissingCat = []

# to be dropped
dropList0 = ["Id", "GarageCars", "Street",  "3SsnPorch", "ScreenPorch"] + BsmtDropped

imputed_on = ["GarageYrBlt", "YearRemodAdd", "OverallQual"]




#######################
dropList = dropList0 +   ["SalePrice"]
fillZeroCont = fillZeroCont + BsmtDropped + imputed_on
#######################

imputeDict = {"Electrical": "SBrkr",
              "Functional": "Typ",
              "Exterior1st": "VinylSd",
              "Exterior2nd": "VinylSd",
              "SaleType": "WD",
              "MSZoning": "RL"}


##########################################################################################################

## Transformer class for pipeline, imputing dict values on NAs
class dictImputer(BaseEstimator, TransformerMixin):
    def __init__(self, dict_: dict):
        self.dict_ = dict_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for k, v in self.dict_.items():
            X[k] = X[k].fillna(v)
        return X[list(self.dict_.keys())]


def make_pipeX():
    nonePipeline = make_pipeline(SimpleImputer(
        strategy="constant", fill_value="None"), OneHotEncoder(drop="first"))
    zeroPipeline = make_pipeline(SimpleImputer(
        strategy="constant", fill_value=0), OneHotEncoder(drop="first", categories="auto"))
    scalePipeline = make_pipeline(SimpleImputer(
        strategy="constant", fill_value=0))#, StandardScaler())

    regressionPipeline = ColumnTransformer([
        ("setNone", nonePipeline, fillNone),
        ("setZero", zeroPipeline, fillZeroCat),
        ("transformed", scalePipeline, fillZeroCont),
        ("dictImputed", make_pipeline(dictImputer(imputeDict),
                                        OneHotEncoder(drop="first")), list(imputeDict.keys())),
        ("bool", "passthrough", imputeBool),
        ("categoricalInts", "passthrough", cat_to_int),
        # ("selected","passthrough",selected),
        ("dropped", "drop", dropList)
    ], remainder="drop")
    return regressionPipeline


pipeline_X = make_pipeX()
pipeline_y = StandardScaler()

train_X = pipeline_X.fit_transform(imputeVals(trainData))
train_y = pipeline_y.fit_transform(np.log(trainData.SalePrice.values.reshape(-1, 1))).reshape(-1,)

test_X = pipeline_X.transform(imputeVals(testData))


search_params = {
    "num_boost_round": [50,100,250],
    "max_depth": [1000],
    "learning_rate": np.linspace(0.01,0.2,3),
    "num_leaves": [5,25,50],
    "min_child_samples": [5,10,20],
    "min_child_weight": [1e-3],
    "subsample": [0.5,0.7,1.0],  # subsample = bagging
    "subsample_freq": [5],
    "colsample_bytree": [0.7,0.85,1.0],  #feature fraction: "mtry"
    #"max_delta_step": -1,
    "reg_alpha": np.logspace(-3,1,5),      # L1
    "reg_lambda": np.logspace(-3,1,5)#,   # L2 
    # "min_split_gain": 0.0,
    # "drop_rate": 0.1, # dart only
    # "max_drop": 50, # dart only
    # "skip_drop": 0.5, # dart only
    # "uniform_drop": False, # dart only
    # "top_rate": 0.2, # goss only
    # "other_rate": 0.1, # goss only
    # "min_data_per_group": 100,
    # "max_cat_threshold": 32,
    # "cat_l2": 10.0,
    # "cat_smooth": 10.0,
    # "max_cat_to_onehot": 4,
    # "topk": 20, # larger -> more accurate but slow
}

search_params = {
    "num_boost_round": [50,100,250],
    "max_depth": [500,750,1000],
    "learning_rate": [0.05,0.1],
    "num_leaves": [5],
    "min_child_samples": [10],
    "min_child_weight": [1e-3],
    "subsample": [0.7],  # subsample = bagging
    "subsample_freq": [5],
    "colsample_bytree": [0.7,0.85,1.0],  #feature fraction: "mtry"
    #"max_delta_step": -1,
    "reg_alpha": [0.05,0.1],      # L1
    "reg_lambda": [2,6,10]#,   # L2 
    # "min_split_gain": 0.0,
    # "drop_rate": 0.1, # dart only
    # "max_drop": 50, # dart only
    # "skip_drop": 0.5, # dart only
    # "uniform_drop": False, # dart only
    # "top_rate": 0.2, # goss only
    # "other_rate": 0.1, # goss only
    # "min_data_per_group": 100,
    # "max_cat_threshold": 32,
    # "cat_l2": 10.0,
    # "cat_smooth": 10.0,
    # "max_cat_to_onehot": 4,
    # "topk": 20, # larger -> more accurate but slow
}

search_params = {
    "num_boost_round": [250,500],
    "max_depth": [250,500,750],
    "learning_rate": [0.05,0.1],
    "num_leaves": [5],
    "min_child_samples": [10],
    "min_child_weight": [1e-3],
    "subsample": [0.7],  # subsample = bagging
    "subsample_freq": [5],
    "colsample_bytree": [0.5,0.7],  #feature fraction: "mtry"
    #"max_delta_step": -1,
    "reg_alpha": [0.05],      # L1
    "reg_lambda": [2]#,   # L2 
    # "min_split_gain": 0.0,
    # "drop_rate": 0.1, # dart only
    # "max_drop": 50, # dart only
    # "skip_drop": 0.5, # dart only
    # "uniform_drop": False, # dart only
    # "top_rate": 0.2, # goss only
    # "other_rate": 0.1, # goss only
    # "min_data_per_group": 100,
    # "max_cat_threshold": 32,
    # "cat_l2": 10.0,
    # "cat_smooth": 10.0,
    # "max_cat_to_onehot": 4,
    # "topk": 20, # larger -> more accurate but slow
}

# verbosity 0 warnings
#          <0 fatal
# lgbm = lgb.LGBMRegressor(objective= "regression", metric="mse", boosting_type="gbdt", device_type = "cpu", tree_learner = "feature", verbosity=1)

# lgb_CV = GridSearchCV(lgbm,search_params,cv=3)
# lgb_CV.fit(train_X, train_y)
# print(lgb_CV.best_params_)
# print(lgb_CV.best_score_)

# fit_params = lgb_CV.best_params_
# fit_params = {'colsample_bytree': 0.7, 'learning_rate': 0.2, 'max_depth': 1000, 'min_child_samples': 10, 'min_child_weight': 0.001, 'num_boost_round': 250, 'num_leaves': 5, 'reg_alpha': 0.01, 'reg_lambda': 1.0, 'subsample': 0.5, 'subsample_freq': 5}   
#       
# fit_params = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 1000, 'min_child_samples': 10, 'min_child_weight': 0.001, 'num_boost_round': 250, 'num_leaves': 5, 'reg_alpha': 0.05, 'reg_lambda': 2, 'subsample': 0.7, 'subsample_freq': 5}  
# Mean squared error: 0.07779883510072638 
# fit_params = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 500, 'min_child_samples': 10, 'min_child_weight': 0.001, 'num_boost_round': 250, 'num_leaves': 5, 'reg_alpha': 0.05, 'reg_lambda': 2, 'subsample': 0.7, 'subsample_freq': 5} 
# Mean squared error: 0.07779883510072638 
# fit_params = {'colsample_bytree': 0.7, 'learning_rate': 0.05, 'max_depth': 250, 'min_child_samples': 10, 'min_child_weight': 0.001, 'num_boost_round': 500, 'num_leaves': 5, 'reg_alpha': 0.05, 'reg_lambda': 2, 'subsample': 0.7, 'subsample_freq': 5} 
# Mean squared error: 0.0747398772357607

fit_params = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 1000, 'min_child_samples': 10, 'min_child_weight': 0.001, 'num_boost_round': 250, 'num_leaves': 5, 'reg_alpha': 0.05, 'reg_lambda': 5, 'subsample': 0.7, 'subsample_freq': 5}  
# Mean squared error: 0.08018601380284113  
# fit_params = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 1000, 'min_child_samples': 10, 'min_child_weight': 0.001, 'num_boost_round': 250, 'num_leaves': 5, 'reg_alpha': 0.1, 'reg_lambda': 7.5, 'subsample': 0.7, 'subsample_freq': 5}  

lgbm = lgb.LGBMRegressor(objective= "regression", metric="mse", boosting_type="dart", device_type = "cpu", tree_learner = "feature", verbosity=1, **fit_params)
lgbm.set_params(**fit_params)
lgbm.fit(train_X,train_y)


train_preds = lgbm.predict(train_X).reshape(-1,)
lgbm_preds = lgbm.predict(test_X)
preds = np.exp(pipeline_y.inverse_transform(lgbm_preds))

print(f'Mean squared error: {mse(train_y,train_preds)}')




submit_frame = pd.DataFrame()
submit_frame['Id'] = testData.Id
submit_frame['SalePrice'] = preds
submit_frame.to_csv('submission.csv', index=False)

pred_frame = pd.DataFrame()
pred_frame['Id'] = trainData.Id
pred_frame['SalePrice'] = np.exp(pipeline_y.inverse_transform(lgbm.predict(train_X)))
pred_frame.to_csv("train_preds.csv",index=False)