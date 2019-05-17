

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