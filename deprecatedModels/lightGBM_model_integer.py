from statsmodels.api import OLS
import statsmodels.api as sm
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

import re

# Disable warnings
pd.set_option('mode.chained_assignment', None)


trainData = pd.read_csv("./data/train.csv")
testData = pd.read_csv("./data/test.csv")

s = "MSZoning, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, SaleType, SaleCondition"

categories = re.split(",\s?",s)

toInt = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
         'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

lgb_cats = [i for i in categories if i not in toInt] + ["MSSubClass", "MoSold"]
print(lgb_cats)

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

    df.Condition1[df.Condition1 == "RRNe"] = "RRNn"
    df.OverallCond[df.OverallCond < 3] = 3
    df.Exterior1st[df.Exterior1st == "AsphShn"] = "AsbShng"
    df.Exterior1st[df.Exterior1st == "ImStucc"] = "Stone"
    df.Heating[df.Heating == "Floor"] = "Grav"
    df.Heating[df.Heating == "OthW"] = "Wall"

    toInt = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
             'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    intDict = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    for i in toInt:
        df[i] = df[i].apply(lambda x: intDict.get(x, 0))

    s = "MSZoning, Street, Alley, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2, BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir, Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish, GarageQual, GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, SaleType, SaleCondition"
    categories = re.split(",\s?", s)
    categories = [i for i in categories if not (i in toInt)]
    for i in categories:
        df[i], _ = pd.factorize(df[i])
    
    #df["CentralAir"] = (df["CentralAir"] == "Y")

    df.LotFrontage = df.LotFrontage.fillna(df.Neighborhood.map(frontage))
    df.MSZoning = df.MSZoning.fillna(df.Neighborhood.map(zoning))
    df.Utilities = df.Utilities.fillna(df.Neighborhood.map(utilities))
    df.MSSubClass[df.MSSubClass == 150] = 120
    df["HasPool"] = df.PoolQC.notnull()

    return df


####################################################################################################
############### THIS IS WHERE YOU SELECT WHICH FEATURES ARE INCLUDED IN THE MODEL ##################
selected = ["sinMonth", ]
# values that null is filled with "None" then get one-hot encoded
fillNone = ["MasVnrType", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageType", "GarageFinish",
            "Fence", "MiscFeature", "MasVnrType", "LotShape", "LandSlope", "Neighborhood",
            "Condition1", "LotConfig", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
            "Foundation", "Heating", "SaleCondition", "Electrical", "PavedDrive"]

# Categorical variables represented as integers
cat_to_int = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
              'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

# ordinal categorical variables
fillZeroCat = ["BsmtFullBath", "HalfBath",
               "MSSubClass", "MoSold", "BsmtHalfBath", ]

 
# continuous variables with missing values that are zero
diffVars = ["garageDiff", "remodDiff"]
fillZeroCont = ["MasVnrArea", "GarageArea", "GrLivArea", "1stFlrSF", "2ndFlrSF", "LotFrontage",
                "TotalBsmtSF", "WoodDeckSF", "OpenPorchSF", "PoolArea", "BedroomAbvGr", "YrSold",
                "LotArea", "EnclosedPorch", "OverallCond", "OverallQual", "MiscVal", "LowQualFinSF",
                "LotFrontage", "FullBath", "Fireplaces", "TotRmsAbvGrd"] + diffVars



# variables that need differences between reference engineered
imputeDiff = [("GarageYrBlt", "YearBuilt"), ("YearRemodAdd", "YearBuilt")]

# imputed to boolean: passthrough
imputeBool = ["CentralAir", "HasPool"]

# categories that we need to know if they were imputed
imputeUnknown = []

# List of values taken out to be onehotencoded with a list argument
# Due to missing values in test data
handleMissingInt = ["FullBath", "GarageCars", "Fireplaces", "TotRmsAbvGrd"]
handleMissingCat = []

# to be dropped
dropList0 = ["TotalBsmtSF", "Id", "GarageCars", "Street", "Alley",
            "Utilities", "Condition2", "3SsnPorch", "ScreenPorch"]

imputed_on = ["GarageYrBlt", "YearRemodAdd", "OverallQual"]

BsmtDropped = ["BsmtFinSF1", "BsmtUnfSF", "BsmtFinSF2"]


#######################
dropList = dropList0 +   ["SalePrice"]
fillZeroCont = fillZeroCont + BsmtDropped + imputed_on
#######################

imputeDict = {"Electrical": "SBrkr",
              "Functional": "Typ",
              "CentralAir": "Y",
              #"KitchenQual": "Fa",
              "SaleType": "Oth",
              "Exterior1st": "VinylSd",
              "Exterior2nd": "VinylSd",
              "SaleType": "WD",
              "MSZoning": "RL"}


##########################################################################################################

## Transformer class for pipeline, imputing dict values on NAs
# class dictImputer(BaseEstimator, TransformerMixin):
#     def __init__(self, dict_: dict):
#         self.dict_ = dict_

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         for k, v in self.dict_.items():
#             X[k] = X[k].fillna(v)
#         return X[self.dict_.keys()]


# def make_pipeX():
#     nonePipeline = make_pipeline(SimpleImputer(
#         strategy="constant", fill_value="None"), OneHotEncoder(drop="first"))
#     zeroPipeline = make_pipeline(SimpleImputer(
#         strategy="constant", fill_value=0), OneHotEncoder(drop="first", categories="auto"))
#     scalePipeline = make_pipeline(SimpleImputer(
#         strategy="constant", fill_value=0), StandardScaler())

#     regressionPipeline = ColumnTransformer([
#         ("setNone", nonePipeline, fillNone),
#         ("setZero", zeroPipeline, fillZeroCat),
#         ("transformed", scalePipeline, fillZeroCont),
#         ("dictImputed", make_pipeline(dictImputer(imputeDict),
#                                         OneHotEncoder(drop="first")), list(imputeDict.keys())),
#         ("bool", "passthrough", imputeBool),
#         ("categoricalInts", "passthrough", cat_to_int),
#         ("selected","passthrough",selected),
#         ("dropped", "drop", dropList)
#     ], remainder="drop")
#     return regressionPipeline


# pipeline_X = make_pipeX()
# pipeline_y = StandardScaler()

# train_X = pipeline_X.fit_transform(imputeVals(trainData))
# train_y = pipeline_y.fit_transform(np.log(trainData.SalePrice.values.reshape(-1, 1)))

# test_X = pipeline_X.transform(imputeVals(testData))

lgbm_params = {'bagging_fraction': 0.1, 'feature_fraction': 0.1, 'lambda': 0.1, 'learning_rate': 0.0325, 'max_bin': 60,
    'metric': 'mse', 'n_estimators': 750, 'num_leaves': 8, 'objective': 'regression', 'sub_feature': 0.5, 'verbose': 0}

lgbm_defaults = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "categorical_features": lgb_cats,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "min_child_weight": 1e-3,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "feature_fraction": 1.0,
    "max_delta_step": -1,
    "lambda_l1": 0.0,
    "lambda_l1": 0.0,
    "min_split_gain": 0.0,
    # "drop_rate": 0.1, # dart only
    # "max_drop": 50, # dart only
    # "skip_drop": 0.5, # dart only
    # "uniform_drop": False, # dart only
    # "top_rate": 0.2, # goss only
    # "other_rate": 0.1, # goss only
    "min_data_per_group": 100,
    "max_cat_threshold": 32,
    "cat_l2": 10.0,
    "cat_smooth": 10.0,
    "max_cat_to_onehot": 4,
    "topk": 20, # larger -> more accurate but slow
}




# lgbm = lgb.LGBMRegressor(metric="mse",**lgbm_defaults)
# lgbm.fit(train_X, train_y)
# lgbm_preds = lgbm.predict(test_X)

# preds = np.exp(pipeline_y.inverse_transform(lgbm_preds))



lgb_data_X = imputeVals(trainData.drop(columns="SalePrice"))
lgb_data_y = np.log(trainData.SalePrice)

lgb_train = lgb.Dataset(lgb_data_X, lgb_data_y)

lgbm = lgb.train(lgbm_defaults, lgb_train)
lgbm.save_model("lgb_modle.txt")

train_preds = lgbm.predict(lgb_data_X)
lgbm_preds = lgbm.predict(imputeVals(testData))
preds = np.exp(lgbm_preds)

print(f'Mean squared error: {mse(lgb_data_y,train_preds)}')


submit_frame = pd.DataFrame()
submit_frame['Id'] = testData.Id
submit_frame['SalePrice'] = preds
submit_frame.to_csv('submission.csv', index=False)
