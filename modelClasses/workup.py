import pandas as pd
import numpy as np
from statsmodels.api import OLS
import statsmodels.api as sm




def getImputeDicts(trainData):
    zoning = trainData.groupby("Neighborhood").MSZoning.apply(
        lambda x: x.value_counts().sort_values().index[0]).to_dict()
    utilities = trainData.groupby("Neighborhood").Utilities.apply(
        lambda x: x.value_counts().sort_values().index[0]).to_dict()
    frontage = trainData.groupby("Neighborhood").LotFrontage.apply(
        lambda x: x.value_counts().sort_values().index[0]).to_dict()

    return {"zoning": zoning, "utilities": utilities, "frontage": frontage}

## Linearize the OverallQual variable

def getQualScale(trainData):
    def testPow(n):
        raw_X = trainData.OverallQual.values.reshape(-1, 1)
        OLS_y = trainData.SalePrice
        X = raw_X**n
        features = sm.add_constant(X)
        ols_sm = OLS(OLS_y.values, features)
        model = ols_sm.fit()
        return model.rsquared
    pws = [testPow(i) for i in np.linspace(2.5, 3.5, 50)]
    qualPow = np.linspace(2.5, 3.5, 50)[np.argmax(pws)]

    return qualPow

