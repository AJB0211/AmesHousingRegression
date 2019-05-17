import pandas as pd
import numpy as np




def nullfill(in_df):
    df = in_df.copy()
    df.Alley = df.Alley.fillna("None")
    df.BsmtQual = df.BsmtQual.fillna("None")
    df.BsmtCond = df.BsmtCond.fillna("None")
    df.BsmtExposure = df.BsmtExposure.fillna("None")
    df.BsmtFinType1 = df.BsmtFinType1.fillna("None")
    df.BsmtFinType2 = df.BsmtFinType2.fillna("None")
    df.FireplaceQu = df.FireplaceQu.fillna("None")
    df.GarageType = df.GarageType.fillna("None")
    df.GarageFinish = df.GarageFinish.fillna("None")
    df.GarageQual = df.GarageQual.fillna("None")
    df.GarageCond = df.GarageCond.fillna("None")
    df.PoolQC = df.PoolQC.fillna("None")
    df.Fence = df.Fence.fillna("None")
    df.MiscFeature = df.MiscFeature.fillna("None")
    df.MasVnrType = df.MasVnrType.fillna("None")
    df.MasVnrArea = df.MasVnrArea.fillna(0)
    df.Electrical = df.Electrical.fillna("SBrkr")
    
    return(df)


zoning = trainData.groupby("Neighborhood").MSZoning.apply(
    lambda x: x.value_counts().sort_values().index[0]).to_dict()
utilities = trainData.groupby("Neighborhood").Utilities.apply(
    lambda x: x.value_counts().sort_values().index[0]).to_dict()
frontage = trainData.groupby("Neighborhood").LotFrontage.apply(
    lambda x: x.value_counts().sort_values().index[0]).to_dict()


# values that null is filled with "None"
fillNone = ["Alley", "BsmtQual", "BsmtCond", "MasVnrType", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual",
            "GarageCond", "PoolQC", "Fence", "MiscFeature", "MasVnrType"]

# For categorical data, it is
fillZero = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "2ndFlrSF", "GrLivArea",
            "MasVnrArea", "GarageArea", "GarageCars", "GarageYrBlt"]


def imputeVals0(in_df):
    df = in_df.copy()
    for i in fillNone:
        df[i] = df[i].fillna("None")
    for i in fillZero:
        # mark which zeros are imputed
        df["null_%s" % (i)] = df[i].isnull()
        df[i] = df[i].fillna(0)
    df.Electrical = df.Electrical.fillna("SBrkr")
    # Documentation instructs to assume "typical" unless otherwise noted
    df.Functional = df.Functional.fillna("Typ")
    df.CentralAir = df.CentralAir.fillna("Y")
    df["null_LotFrontage"] = df.LotFrontage.isnull()
    df.LotFrontage = df.LotFrontage.fillna(0)
    df.MSZoning = df.MSZoning.fillna(df.Neighborhood.map(zoning))
    df.Utilities = df.Utilities.fillna(df.Neighborhood.map(utilities))
    df.KitchenQual = df.KitchenQual.fillna(
        "Po")  # one house missing kitchen data
    # only one missing value, fill the already defined "other"
    df.SaleType = df.SaleType.fillna("Oth")
    df.Exterior1st = df.Exterior1st.fillna("Other")
    # the same house is responsible for the missing exterior 1st and 2nd, other is predefined
    df.Exterior2nd = df.Exterior2nd.fillna("Other")
    df = df.drop(columns=["Id"])
    return(df)


def imputeVals1(in_df):
    df = in_df.copy()
    for i in fillNone:
        df[i] = df[i].fillna("None")
    for i in fillZero:
        df["null_%s" % (i)] = df[i].isnull()
        df[i] = df[i].fillna(0)
    df.Electrical = df.Electrical.fillna("SBrkr")
    df.Functional = df.Functional.fillna("Typ")
    df.CentralAir = df.CentralAir.fillna("Y")
    # This is the only line different in these two functions, maybe a more elegant solution is possible
    df.LotFrontage = df.LotFrontage.fillna(df.Neighborhood.map(frontage))
    df.MSZoning = df.MSZoning.fillna(df.Neighborhood.map(zoning))
    df.Utilities = df.Utilities.fillna(df.Neighborhood.map(utilities))
    df.KitchenQual = df.KitchenQual.fillna("Po")
    df.SaleType = df.SaleType.fillna("Oth")
    df.Exterior1st = df.Exterior1st.fillna("Other")
    df.Exterior2nd = df.Exterior2nd.fillna("Other")
    df = df.drop(columns=["Id"])
    return(df)


normedPrice = np.log(trainData.SalePrice)
sp_mean = np.mean(normedPrice)
sp_std = np.std(normedPrice)
normedPrice = (normedPrice - sp_mean) / sp_std


