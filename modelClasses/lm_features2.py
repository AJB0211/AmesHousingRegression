import pandas as pd
import numpy as np

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
        df["numBaths"] = df.BsmtFullBath + 0.5 * \
            df.BsmtHalfBath + df.FullBath + 0.5*df.HalfBath
        df["bsmtBaths"] = df.BsmtFullBath + 0.5*df.BsmtHalfBath

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
"Foundation", "Heating", "SaleCondition", "Electrical", "PavedDrive"]

# labelEncode = [""]

# Categorical variables represented as integers
cat_to_int = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
              'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC', "YrSold"]

# ordinal categorical variables
fillZeroCat = ["BsmtFullBath", "HalfBath", "MSSubClass"]

# continuous variables with missing values that are zero
fillZeroCont = ["MasVnrArea", "GarageArea", "GrLivArea", "LotFrontage", "1stFlrSF",
    "TotalBsmtSF", "WoodDeckSF", "OpenPorchSF", "PoolArea", "BedroomAbvGr", 
    "LotArea", "EnclosedPorch", "OverallCond", "scaledOverallQual", "SF",
    "squareMo", "quadMo", "hexMo", "numBaths", "bsmtBaths",
    "LotFrontage", "Fireplaces", "TotRmsAbvGrd", "garageDiff", "remodDiff"]

# variables that need differences between reference engineered
imputeDiff = [("GarageYrBlt", "YearBuilt"),
    ("YearRemodAdd", "YearBuilt")]

# imputed to boolean: passthrough
imputeBool = ["CentralAir", "HasPool", "HasBsmt", "is2story"]

# categories that we need to know if they were imputed
imputeUnknown = []

# List of values taken out to be onehotencoded with a list argument
# Due to missing values in test data
handleMissingInt = ["FullBath", "GarageCars",
        "Fireplaces", "TotRmsAbvGrd"]
handleMissingCat = []

# to be dropped
dropList = ["TotalBsmtSF", "Id", "GarageCars", "Street", "Alley",
"Utilities", "Condition2", "LowQualFinSF", "BsmtHalfBath", "3SsnPorch", "ScreenPorch",
"PoolArea", "PoolQC", "MiscVal", "OverallQual"]

newDrops = ["MiscFeature", "BsmtFullBath",
"BsmtHalfBath", "FullBath", "HalfBath",
"1stFlrSF", "2ndFlrSF", ]

drop_imputed = ["MoSold", "GarageYrBlt",
    "YearRemodAdd", "PoolArea", "PoolQC", "OverallQual"]

BsmtDropped = ["BsmtFinSF1", "BsmtUnfSF", "BsmtFinSF2"]

imputeDict = {"Electrical": "Oth",
    "Functional": "Typ",
    "CentralAir": True,
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "SaleType": "WD",
    "MSZoning": "RL"}

dropList = dropList + newDrops + drop_imputed
