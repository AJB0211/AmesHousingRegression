from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np 
import pandas as pd
import sklearn

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV

# Disable warnings
pd.set_option('mode.chained_assignment', None)


trainData = pd.read_csv("./data/train.csv")
testData = pd.read_csv("./data/test.csv")

outliers = ((trainData.GrLivArea > 4000) & (trainData.SalePrice < 5E5))
trainData = trainData[~(outliers)]

group2 = ["ClearCr", "Timber", "NPkVill", "Blueste", "BrDale",
          "MeadowV", "IDOTRR", "SWISU", "Blmngtn", "Mitchel"]
trainData["group"] = trainData.Neighborhood.isin(group2)
testData["group"] = testData.Neighborhood.isin(group2)

trainBoro = trainData.groupby("group")["Id"].apply(np.array).to_dict()
testBoro = testData.groupby("group")["Id"].apply(np.array).to_dict()

trainBoro_dfidx = trainData.reset_index().groupby("group").apply(lambda x: x.index.values).to_dict()
testBoro_dfidx = testData.reset_index().groupby("group").apply(lambda x: x.index.values).to_dict()

zoning = trainData.groupby("Neighborhood").MSZoning.apply(lambda x: x.value_counts().sort_values().index[0]).to_dict()
utilities = trainData.groupby("Neighborhood").Utilities.apply(lambda x: x.value_counts().sort_values().index[0]).to_dict()
frontage = trainData.groupby("Neighborhood").LotFrontage.apply(lambda x: x.value_counts().sort_values().index[0]).to_dict()

## Linearize the OverallCond variable
import statsmodels.api as sm
from statsmodels.api import OLS

def testPow(n):
    raw_X = trainData.OverallQual.values.reshape(-1,1)
    OLS_y = trainData.SalePrice
    X = raw_X**n
    features = sm.add_constant(X)
    ols_sm   = OLS(OLS_y.values,features)
    model    = ols_sm.fit()
    return model.rsquared

pws = [testPow(i) for i in np.linspace(2.5,3.5,50)]
qualPow = np.linspace(2.5,3.5,50)[np.argmax(pws)]

## Function for values that are manually imputed


def imputeVals(in_df):
    df = in_df.copy()
    df.LotFrontage = df.LotFrontage.fillna(df.Neighborhood.map(frontage))
    df.MSZoning = df.MSZoning.fillna(df.Neighborhood.map(zoning))
    df.Utilities = df.Utilities.fillna(df.Neighborhood.map(utilities))
    df["CentralAir"] = (df["CentralAir"] == "Y")
    df.MSSubClass[df.MSSubClass == 150] = 120
    df["sinMonth"] = df.MoSold.apply(lambda x: np.sin(np.pi*x/12))
    df["scaledOverallQual"] = df.OverallQual.apply(lambda x: x**qualPow)
    df.Condition1[df.Condition1 == "RRNe"] = "RRNn"
    df.OverallCond[df.OverallCond < 3] = 3
    df.Exterior1st[df.Exterior1st == "AsphShn"] = "AsbShng"
    df.Exterior1st[df.Exterior1st == "ImStucc"] = "Stone"
    df.ExterCond[df.ExterCond == "Po"] = "Fa"
    df.BsmtCond[df.BsmtCond == "Po"] = np.nan
    df.Heating[df.Heating == "Floor"] = "Grav"
    df.Heating[df.Heating == "OthW"] = "Wall"
    df.HeatingQC[df.HeatingQC == "Po"] = "Fa"
    df.Electrical[df.Electrical != "SBrkr"] = "Oth"
    df["HasPool"] = df.PoolQC.notnull()
    df["garageDiff"] = df.GarageYrBlt - df.YearBuilt
    df["remodDiff"] = df.YearRemodAdd - df.YearBuilt
    return df

####################################################################################################
############### THIS IS WHERE YOU SELECT WHICH FEATURES ARE INCLUDED IN THE MODEL ##################
selected = []
# values that null is filled with "None"
fillNone = ["BsmtQual", "MasVnrType", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtCond", "FireplaceQu",
            "PoolQC", "Fence", "MiscFeature", "MasVnrType", "LotShape", "LandSlope", "Neighborhood",
            "Condition1", "LotConfig", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "ExterQual",
            "ExterCond", "Foundation", "Heating", "HeatingQC", "SaleCondition", "Electrical", "PavedDrive"]

# ordinal categorical variables
fillZeroCat = ["BsmtFullBath", "HalfBath","MSSubClass"]

# continuous variables with missing values that are zero
fillZeroCont = ["MasVnrArea", "GarageArea", "GrLivArea", "1stFlrSF", "2ndFlrSF", "LotFrontage",
                "TotalBsmtSF", "WoodDeckSF", "OpenPorchSF", "PoolArea", "BedroomAbvGr", "YrSold",
                "LotArea", "EnclosedPorch", "sinMonth", "OverallCond", "scaledOverallQual",
                "LotFrontage", "FullBath", "Fireplaces", "TotRmsAbvGrd", "garageDiff","remodDiff"]

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
dropList = ["TotalBsmtSF", "MoSold", "Id", "GarageYrBlt", "YearRemodAdd", "GarageCars", "Street", "Alley",
            "Utilities", "MoSold", "Condition2", "LowQualFinSF", "BsmtHalfBath", "3SsnPorch", "ScreenPorch",
            "PoolArea", "PoolQC", "MiscVal", "OverallQual","group"]

BsmtDropped = ["BsmtFinSF1", "BsmtUnfSF", "BsmtFinSF2"]

imputeDict = {"Electrical": "Oth",
              "Functional": "Typ",
              "CentralAir": "Y",
              "KitchenQual": "Fa",
              "SaleType": "Oth",
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
        return X[self.dict_.keys()]


## Pipeline construction
nonePipeline = make_pipeline(SimpleImputer(
    strategy="constant", fill_value="None"), OneHotEncoder(drop="first"))
zeroPipeline = make_pipeline(SimpleImputer(
    strategy="constant", fill_value=0), OneHotEncoder(drop="first", categories="auto"))
scalePipeline = make_pipeline(SimpleImputer(
    strategy="constant", fill_value=0), StandardScaler())

## Pipeline for X 
regressionPipeline = ColumnTransformer([
    ("setNone", nonePipeline, fillNone),
    ("setZero", zeroPipeline, fillZeroCat),
    ("transformed", scalePipeline, fillZeroCont),
    ("dictImputed", make_pipeline(dictImputer(imputeDict),
                                  OneHotEncoder(drop="first")), list(imputeDict.keys())),
    ("bool", "passthrough", imputeBool),
    ("dropped", "drop", dropList)
], remainder="drop")

## Pipeline for y
targetScaler = StandardScaler()

## Fit and transform X and y in train set
train_X = trainData.drop(columns=["SalePrice"])
train_y = trainData.SalePrice
pipe_X = imputeVals(train_X)


########################################################################
## NOTE THIS INPUT STATEMENT
# print(pipe_X.group.value_counts())
# print(pipe_X.sinMonth)
# input("Check values before continuing")
########################################################################

piped_X = regressionPipeline.fit_transform(pipe_X)
pipe_y = targetScaler.fit_transform(np.log(train_y.values.reshape(-1, 1)))


## Borough regression class
class boroReg:
    def __init__(self, X, y, idx, pipe_X, pipe_y):
        self.X = X[idx, :]  # shift to fix 1 indexing using np broadcasting
        self.y = y[idx, :]
        self._gridSearch = None
        self.pipeline_X = pipe_X
        self.pipeline_y = pipe_y
        self._searchSpace = None
        self._params = None
        self.lm = ElasticNet()

    def __imputeVals(self, in_df):
        return imputeVals(in_df)

    def gridSearch(self, params, cv=5, njobs=-1, verbose=50):
        self._searchSpace = params

        self._gridSearch = GridSearchCV(
            self.lm, params, cv=cv, scoring="neg_mean_squared_error", n_jobs=njobs, verbose=verbose)
        self._gridSearch.fit(self.X, self.y)

    def getBestParams(self):
        if self._gridSearch is not None:
            return self._gridSearch.best_params_
        else:
            raise ValueError()

    def getBestScore(self):
        if self._gridSearch is not None:
            return self._gridSearch.best_score_
        else:
            raise ValueError()

    def fitModel(self, params):
        self._params = params

        self.lm.set_params(**params)
        self.lm.fit(self.X, self.y)

    def __invert(self, y):
        return np.exp(self.pipeline_y.inverse_transform(y))

    def getTrainScore(self):
        return self.lm.score(self.X, self.y)

    def predict(self, test_X):
        piped_X = self.pipeline_X.transform(self.__imputeVals(test_X))
        preds = self.lm.predict(piped_X)
        return self.__invert(preds)


def perBoroSearch(id_list, X, y, params, pipe_X, pipe_y):
    local_lm = boroReg(X, y, id_list, pipe_X, pipe_y)
    local_lm.gridSearch(params)
    return local_lm


def makeBoroDict(dict_, X, y, pipe_X, pipe_y):
    return {i: boroReg(X, y, j, pipe_X, pipe_y) for i, j in dict_.items()}


def boroPred(y, idx, boroClass):
    boroClass.predict(y[idx, :])

### Parameter search, comment out when values are known
# search5 = {"alpha": np.logspace(-5, 2, 20), "l1_ratio": np.linspace(0, 1, 20), "max_iter": np.array([10000])}
# boroDict = {i: perBoroSearch(j, piped_X, pipe_y, search5, regressionPipeline, targetScaler) for i, j in trainBoro_dfidx.items()}
# boroParams = {i: j.getBestParams() for i, j in boroDict.items()}
# for j in boroParams.values():
#     j["max_iter"] = 100000

### For when parameters are known
boroDict = makeBoroDict(trainBoro_dfidx,piped_X,pipe_y,regressionPipeline,targetScaler)
boroParams = {False: {'alpha': 0.008858667904100823, 'l1_ratio': 0.05263157894736842, 'max_iter': 100000}, True: {
    'alpha': 0.02069138081114788, 'l1_ratio': 0.3684210526315789, 'max_iter': 100000}}

## Fitting to best parameters in search
for i, j in boroDict.items():
    j.fitModel(boroParams[i])

## Making predictions for each neighborhood grouping on test data
outPreds = {i: j.predict(testData.iloc[testBoro_dfidx[i], :]) for i, j in boroDict.items()}
outTupleList = ([(k, l) for i, j in zip(testBoro.values(),outPreds.values()) for k, l in zip(i, j)])
outSeries = pd.Series(dict(outTupleList))

## Creating submission csv file
submit_frame = pd.DataFrame()
submit_frame['Id'] = outSeries.index
submit_frame['SalePrice'] = outSeries.values
submit_frame.to_csv('submission.csv', index=False)

boroMSE = {i: j.getTrainScore() for i, j in boroDict.items()}
print(boroMSE)
print(boroParams)

print(outSeries.shape)
