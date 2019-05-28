import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

pd.set_option('mode.chained_assignment', None)


## Function for values that are manually imputed
## THIS IS A CURRIED FUNCTION
def impute_shell(frontage, zoning, utilities,qualPow):
    def imputeVals(in_df):
        df = in_df.copy()
        df.LotFrontage = df.LotFrontage.fillna(
            df.Neighborhood.map(frontage))
        df.MSZoning = df.MSZoning.fillna(df.Neighborhood.map(zoning))
        df.Utilities = df.Utilities.fillna(
            df.Neighborhood.map(utilities))
        df["CentralAir"] = (df["CentralAir"] == "Y")
        df.MSSubClass[df.MSSubClass == 150] = 120
        df.RoofMatl[df.RoofMatl.isin(
            ["ClyTile", "Membran", "Metal", "Roll"])] = "Oth"
        #df["sinMonth"] = df.MoSold.apply(lambda x: np.sin(np.pi*x/12))
        df["scaledOverallQual"] = df.OverallQual.apply(
            lambda x: x**qualPow)
        df.Condition1[df.Condition1 == "RRNe"] = "RRNn"
        df.OverallCond[df.OverallCond < 3] = 3
        df.Exterior1st[df.Exterior1st == "AsphShn"] = "AsbShng"
        df.Exterior1st[df.Exterior1st == "ImStucc"] = "Stone"
        #df.ExterCond[df.ExterCond == "Po"] = "Fa"
        #df.BsmtCond[df.BsmtCond == "Po"] = np.nan
        df.Heating[df.Heating == "Floor"] = "Grav"
        df.Heating[df.Heating == "OthW"] = "Wall"
        #df.HeatingQC[df.HeatingQC == "Po"] = "Fa"
        df.Electrical[df.Electrical != "SBrkr"] = "Oth"
        df["HasPool"] = df.PoolQC.notnull()
        df["garageDiff"] = df.GarageYrBlt - df.YearBuilt
        df["remodDiff"] = df.YearRemodAdd - df.YearBuilt
        df["squareMo"] = df.MoSold ** 2
        df["quadMo"] = df.MoSold ** 4
        df["hexMo"] = df.MoSold ** 6
        df["octMo"] = df.MoSold ** 8

        toInt = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
        intDict = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
        for i in toInt:
            df[i] = df[i].apply(lambda x: intDict.get(x, 0))

        df["SF"] = df.TotalBsmtSF + df["1stFlrSF"] + df["2ndFlrSF"]
        df["HasBsmt"] = (df.TotalBsmtSF != 0)
        df["is2story"] = (df["2ndFlrSF"] != 0)
        df["numBaths"] = df.BsmtFullBath + 0.5*df.BsmtHalfBath + df.FullBath + 0.5*df.HalfBath
        df["bsmtBaths"] = df.BsmtFullBath + 0.5*df.BsmtHalfBath
        df["newHouse"] = (df.YearBuilt == df.YrSold)

        return df
    
    return imputeVals

##########################################################################################################

####################################################################################################
############### THIS IS WHERE YOU SELECT WHICH FEATURES ARE INCLUDED IN THE MODEL ##################
selected = []
# values that null is filled with "None" then get one-hot encoded
fillNone = ["MasVnrType", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageType", "GarageFinish",
            "Fence",  "MasVnrType", "LotShape", "LandSlope", "Neighborhood",
            "Condition1", "LotConfig", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
            "Foundation", "Heating", "SaleCondition", "PavedDrive"]

fillNoneLabelEnc = [
    # "Street", 
    # "Alley", 
    # "Utilities"
    ]
# Currently not cooperating

# Categorical variables represented as integers
pass_cats = ["numBaths", "bsmtBaths", "FullBath", "GarageCars", "Fireplaces",
             "BsmtFullBath", "BsmtHalfBath", "HalfBath",
             "Fireplaces", "TotRmsAbvGrd", "OverallCond"]

cat_to_int = (['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
              'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', "YrSold"] 
              + pass_cats
            )

# ordinal categorical variables
fillZeroCat = ["MSSubClass"]

# Due to missing values in test data


# continuous variables with missing values that are zero
diff_vars = ["garageDiff", "remodDiff"]
monthVars = ["squareMo", "quadMo", "hexMo"]
fillZeroCont = ([
    "MasVnrArea", "GarageArea", "GrLivArea", "LotFrontage", "1stFlrSF", "2ndFlrSF",
    "TotalBsmtSF", "WoodDeckSF", "OpenPorchSF", "PoolArea", "BedroomAbvGr", 
    "EnclosedPorch", "scaledOverallQual", "SF", "LotFrontage", "LowQualFinSF",
    "MiscVal", "OverallQual", "3SsnPorch", "ScreenPorch", "GarageYrBlt",
    # "YearRemodAdd", CAUSES OVERFLOW
    "BsmtFinSF1", "BsmtUnfSF", "BsmtFinSF2"
    ] 
     + monthVars 
    )



lotAreaCustom = ["LotArea"]

# variables that need differences between reference engineered
imputeDiff = [("GarageYrBlt", "YearBuilt"),
    ("YearRemodAdd", "YearBuilt")]

# imputed to boolean: passthrough
imputeBool = ["CentralAir", "HasPool", "HasBsmt", "is2story","newHouse"]

# categories that we need to know if they were imputed
imputeUnknown = []

# List of values taken out to be onehotencoded with a list argument

handleMissingCat = []

# to be dropped
dropList = ["Id", "Condition2", "MiscFeature", "MoSold"]


imputeDict = {"Electrical": "Oth",
    "Functional": "Typ",
    "CentralAir": True,
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "SaleType": "WD",
    # "MSZoning": "RL"
    }
    

def make_pipe():
        #import lm_features as f
        nonePipeline = make_pipeline(
            SimpleImputer(strategy="constant", fill_value="None"), 
            OneHotEncoder(drop="first"))
        zeroPipeline = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0), 
            OneHotEncoder(drop="first", categories="auto"))
        scalePipeline = make_pipeline(
                            SimpleImputer(strategy="constant", fill_value=0), 
                            PowerTransformer(standardize=False), 
                            RobustScaler())

        regressionPipeline = ColumnTransformer([
            ("setNone", nonePipeline, fillNone),
            # ("labeled", make_pipeline(
            #     SimpleImputer(strategy="constant",fill_value="None"),
            #     LabelEncoder()), fillNoneLabelEnc),

            ("labeled", SimpleImputer(strategy="constant",fill_value="None"), fillNoneLabelEnc),
            ("setZero", zeroPipeline, fillZeroCat),
            ("transformed", scalePipeline, fillZeroCont),
            ("dictImputed", make_pipeline(dictImputer(imputeDict),
                                          OneHotEncoder(drop="first")), list(imputeDict.keys())),
            ("lotArea", SimpleImputer(strategy="constant",fill_value=0),lotAreaCustom),
            ("diffVars", make_pipeline(SimpleImputer(strategy="constant", fill_value=0),StandardScaler()), diff_vars),
            ("bool", "passthrough", imputeBool),
            ("categoricalInts", SimpleImputer(strategy="constant",fill_value=0), cat_to_int)
        ], remainder="drop")
        return regressionPipeline


class dictImputer(BaseEstimator, TransformerMixin):
    def __init__(self, dict_: dict):
        self.dict_ = dict_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for k, v in self.dict_.items():
            X[k] = X[k].fillna(v)
        return X[self.dict_.keys()]
