import numpy as np 
import pandas as pd

#import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

# Disable warnings
pd.set_option('mode.chained_assignment', None)


trainData = pd.read_csv("./data/train.csv")
testData = pd.read_csv("./data/test.csv")

outliers = ((trainData.GrLivArea > 4000) & (trainData.SalePrice < 5E5))
trainData = trainData[~(outliers)]

zoning = trainData.groupby("Neighborhood").MSZoning.apply(lambda x: x.value_counts().sort_values().index[0]).to_dict()
utilities = trainData.groupby("Neighborhood").Utilities.apply(lambda x: x.value_counts().sort_values().index[0]).to_dict()
frontage = trainData.groupby("Neighborhood").LotFrontage.apply(lambda x: x.value_counts().sort_values().index[0]).to_dict()

## Linearize the OverallQual variable
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
    #df["sinMonth"] = df.MoSold.apply(lambda x: np.sin(np.pi*x/12))
    df["scaledOverallQual"] = df.OverallQual.apply(lambda x: x**qualPow)
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
        df[i] = df[i].apply(lambda x: intDict.get(x,0))

    df["SF"] = df.TotalBsmtSF + df["1stFlrSF"] + df["2ndFlrSF"]
    df["HasBsmt"] = (df.TotalBsmtSF != 0)
    df["is2story"] = (df["2ndFlrSF"] != 0)
    df["numBaths"] = df.BsmtFullBath + 0.5*df.BsmtHalfBath + df.FullBath + 0.5*df.HalfBath
    df["bsmtBaths"] = df.BsmtFullBath + 0.5*df.BsmtHalfBath

    return df

####################################################################################################
############### THIS IS WHERE YOU SELECT WHICH FEATURES ARE INCLUDED IN THE MODEL ##################
selected = []
# values that null is filled with "None" then get one-hot encoded
fillNone = ["MasVnrType", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageType", "GarageFinish",
            "Fence",  "MasVnrType", "LotShape", "LandSlope", "Neighborhood",
            "Condition1", "LotConfig", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", 
             "Foundation", "Heating", "SaleCondition", "Electrical", "PavedDrive"]

# Categorical variables represented as integers
cat_to_int = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
              'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

# ordinal categorical variables
fillZeroCat = ["BsmtFullBath", "HalfBath","MSSubClass"]

# continuous variables with missing values that are zero
fillZeroCont = ["MasVnrArea", "GarageArea", "GrLivArea", "LotFrontage", "1stFlrSF",
                "TotalBsmtSF", "WoodDeckSF", "OpenPorchSF", "PoolArea", "BedroomAbvGr", "YrSold",
                "LotArea", "EnclosedPorch", "OverallCond", "scaledOverallQual", "SF",
                "squareMo", "quadMo", "hexMo", "octMo", "numBaths", "bsmtBaths",
                "LotFrontage", "Fireplaces", "TotRmsAbvGrd", "garageDiff","remodDiff"]

# variables that need differences between reference engineered
imputeDiff = [("GarageYrBlt", "YearBuilt"), ("YearRemodAdd", "YearBuilt")]

# imputed to boolean: passthrough
imputeBool = ["CentralAir", "HasPool", "HasBsmt", "is2story"]

# categories that we need to know if they were imputed
imputeUnknown = []

# List of values taken out to be onehotencoded with a list argument
# Due to missing values in test data
handleMissingInt = ["FullBath", "GarageCars", "Fireplaces", "TotRmsAbvGrd"]
handleMissingCat = []

# to be dropped
dropList = ["TotalBsmtSF", "Id", "GarageCars", "Street", "Alley",
            "Utilities", "Condition2", "LowQualFinSF", "BsmtHalfBath", "3SsnPorch", "ScreenPorch",
            "PoolArea", "PoolQC", "MiscVal", "OverallQual"]

newDrops = ["MiscFeature", "BsmtFullBath",
            "BsmtHalfBath", "FullBath", "HalfBath",
            "1stFlrSF", "2ndFlrSF", ]



drop_imputed = ["MoSold", "GarageYrBlt", "YearRemodAdd", "PoolArea","PoolQC","OverallQual"]

BsmtDropped = ["BsmtFinSF1", "BsmtUnfSF", "BsmtFinSF2"]

imputeDict = {"Electrical": "Oth",
              "Functional": "Typ",
              "CentralAir": True,
              "Exterior1st": "VinylSd",
              "Exterior2nd": "VinylSd",
              "SaleType": "WD",
              "MSZoning": "RL"}

dropList = dropList + newDrops + drop_imputed


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

## Class for bundling linear regression operations
class linReg:
    def __init__(self, in_df):
        df = self.__imputeVals(in_df.copy())
        self.X = df.drop(columns=["SalePrice"]).copy()
        self.y = np.log(df.SalePrice.values.reshape(-1, 1))

        self._gridSearch = None
        self.pipeline_X = self.__make_pipe()
        self.pipeline_y = StandardScaler()
        self._searchSpace = None
        self._params = None
        self.lm = ElasticNet()

    def __imputeVals(self, in_df):
        return imputeVals(in_df)

    def __make_pipe(self):
        nonePipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value="None"), OneHotEncoder(drop="first"))
        zeroPipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value=0), OneHotEncoder(drop="first", categories="auto"))
        scalePipeline = make_pipeline(SimpleImputer(
            strategy="constant", fill_value=0), StandardScaler())

        regressionPipeline = ColumnTransformer([
            ("setNone", nonePipeline, fillNone),
            ("setZero", zeroPipeline, fillZeroCat),
            ("transformed", scalePipeline, fillZeroCont),
            ("dictImputed", make_pipeline(dictImputer(imputeDict),
                                          OneHotEncoder(drop="first")), list(imputeDict.keys())),
            ("bool", "passthrough", imputeBool),
            ("categoricalInts", "passthrough", cat_to_int),
            ("dropped", "drop", dropList)
        ], remainder="drop")
        return regressionPipeline

    def gridSearch(self, params, cv=5, njobs=-1, verbose=50):
        self._searchSpace = params
        #self._params = None

        piped_X = self.pipeline_X.fit_transform(self.X)
        piped_y = self.pipeline_y.fit_transform(self.y)
        self._gridSearch = GridSearchCV(
            self.lm, params, cv=cv, scoring="neg_mean_squared_error", n_jobs=njobs, verbose=verbose)
        self._gridSearch.fit(piped_X, piped_y)

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
        piped_X = self.pipeline_X.fit_transform(self.X)
        piped_y = self.pipeline_y.fit_transform(self.y)
        self._params = params

        self.lm.set_params(**params)
        self.lm.fit(piped_X, piped_y)

    def __invert(self, y):
        return np.exp(self.pipeline_y.inverse_transform(y))

    def getTrainScore(self):
        piped_X = self.pipeline_X.transform(self.X)
        piped_y = self.pipeline_y.transform(self.y)
        return self.lm.score(piped_X, piped_y)

    def predict(self, test_X):
        piped_X = self.pipeline_X.transform(self.__imputeVals(test_X))
        preds = self.lm.predict(piped_X)
        return self.__invert(preds)


search1 = {"alpha": np.logspace(-5, 2, 25), "l1_ratio": np.linspace(0, 1, 25)}#, "max_iter":np.array([10000])}
lm = linReg(trainData)

### For grid search
# lm.gridSearch(search1)
# print("\n\n\n")
# print("="*25)
# print(f'GridSearch score: {lm.getBestScore()}\n\n')
# params = lm.getBestParams()
# params["max_iter"] = 1e6
# print(params)

params = {'alpha': 0.0005623413251903491, 'l1_ratio': 1.0}
lm.fitModel(params)
print(f'Train score: {lm.getTrainScore()}')

preds = lm.predict(testData)

submit_frame = pd.DataFrame()
submit_frame['Id'] = testData.Id
submit_frame['SalePrice'] = preds
submit_frame.to_csv('submission.csv', index=False)

train_pred_frame = pd.DataFrame()
train_pred_frame["Id"] = trainData.Id
train_pred_frame["SalePrice"] = lm.predict(trainData)
train_pred_frame.to_csv("train_preds.csv")

# {'alpha': 0.0005623413251903491, 'l1_ratio': 1.0}

# {'alpha': 0.0005179474679231213, 'l1_ratio': 1.0} #0.13166
