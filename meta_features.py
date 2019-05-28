import pandas as pd
import numpy as np

pd.set_option('mode.chained_assignment', None)


## Function for values that are manually imputed
## THIS IS A CURRIED FUNCTION
def impute_shell(qualPow):
    def imputeVals(in_df):
        df = in_df.copy()
        df["scaledOverallQual"] = df.OverallQual.apply(
            lambda x: x**qualPow)
        
        toInt = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
        intDict = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
        for i in toInt:
            df[i] = df[i].apply(lambda x: intDict.get(x, 0))

        df["SF"] = df.TotalBsmtSF + df["1stFlrSF"] + df["2ndFlrSF"]
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
fillNone = ["GarageFinish","Neighborhood"]
            
# Categorical variables represented as integers
cat_to_int = ['ExterQual', 'BsmtQual', 'GarageQual']

# ordinal categorical variables
fillZeroCat = []

# continuous variables with missing values that are zero
fillZeroCont = ["MasVnrArea", "GarageArea", "GrLivArea", "TotalBsmtSF", 
                "scaledOverallQual", "SF","numBaths","TotRmsAbvGrd",]

imputeDict = {"Electrical": "Oth",
              "Functional": "Typ",
              "CentralAir": True,
              "Exterior1st": "VinylSd",
              "Exterior2nd": "VinylSd",
              "SaleType": "WD",
              "MSZoning": "RL"}

