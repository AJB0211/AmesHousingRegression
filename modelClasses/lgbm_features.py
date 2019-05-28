import pandas as pd
import numpy as np

pd.set_option('mode.chained_assignment', None)

####################################################################################################
############### THIS IS WHERE YOU SELECT WHICH FEATURES ARE INCLUDED IN THE MODEL ##################
selected = ["sinMonth"]
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
rmVars1 = ["MiscVal", "LowQualFinSF",
           "BsmtFullBath", "HalfBath", "BsmtHalfBath"]
fillZeroCont = ["MasVnrArea", "GarageArea", "GrLivArea", "1stFlrSF", "2ndFlrSF", "LotFrontage",
                "TotalBsmtSF", "WoodDeckSF", "OpenPorchSF", "PoolArea", "BedroomAbvGr", "YrSold", "MoSold",
                "LotArea", "EnclosedPorch", "OverallCond", "OverallQual",
                "LotFrontage", "FullBath", "Fireplaces", "TotRmsAbvGrd", "TotalBsmtSF", "numBaths", "Fireplaces", "TotRmsAbvGrd"] + rmVars1

# variables that need differences between reference engineered
imputeDiff = [("GarageYrBlt", "YearBuilt"), ("YearRemodAdd", "YearBuilt")]

# imputed to boolean: passthrough
imputeBool = ["CentralAir", "HasPool", "2story", "hasBsmt"]

# categories that we need to know if they were imputed
imputeUnknown = []

# List of values taken out to be onehotencoded with a list argument
# Due to missing values in test data
handleMissingInt = ["GarageCars"]
handleMissingCat = []

# to be dropped
dropList0 = ["Id", "GarageCars", "Street",
             "3SsnPorch", "ScreenPorch"] + BsmtDropped

imputed_on = ["GarageYrBlt", "YearRemodAdd", "OverallQual"]


#######################
dropList = dropList0
fillZeroCont = fillZeroCont + BsmtDropped + imputed_on
#######################

imputeDict = {"Electrical": "SBrkr",
              "Functional": "Typ",
              "Exterior1st": "VinylSd",
              "Exterior2nd": "VinylSd",
              "SaleType": "WD",
              "MSZoning": "RL"}


## THIS IS A CURRIED FUNCTION
def imputeShell(frontage,zoning,utilities):
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
        df["numBaths"] = df.BsmtFullBath + 0.5 * \
            df.BsmtHalfBath + df.FullBath + 0.5*df.HalfBath
        df["2story"] = (df["2ndFlrSF"] != 0)
        df["hasBsmt"] = (df["TotalBsmtSF"] != 0)

        return df

    return imputeVals
