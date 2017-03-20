import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


class FeatureEnge:
    '''
    Do some feature engeneering on dataset
    '''

    def __init__(self, train, test):
        '''
        concat testset and trainingset into X
        '''
        self.X = pd.concat([train.iloc[:, 1: -1], test.iloc[:, 1:]])
        self.ytrain = train.iloc[:, -1]
        self.train_shape = train.shape
        self.test_shape = test.shape

    def drop(self):
        '''
        drop columns which are useless or cause overfitting
            Street is less in testset
            Fence, PoolQC is less in dataset
            MiscFeature can be replaced by MiscVal
        '''
        self._droplist = ["Street", "PoolQC", "Fence", "MiscFeature"]
        self.X.drop(self._droplist, axis=1, inplace=True)

    def new_features(self):
        '''
        Add new features in x
        Built year and sold year may have relationship

        '''
        self.X.loc[:, "YrBuilt_Sold"] = pd.Series(
            self.X.loc[:, "YrSold"].values - self.X.loc[:, "YearBuilt"].values)
        # using numbers instead of catelogical fearutes
        self.qual_dict = {None: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        self.qual_list = ['ExterQual',
                          'ExterCond',
                          'BsmtQual',
                          'BsmtCond',
                          'HeatingQC',
                          'KitchenQual',
                          'FireplaceQu',
                          'GarageQual',
                          'GarageCond']
        for series in self.qual_list:
            self.X[series] = pd.to_numeric(self.X[series].map(self.qual_dict), errors="coerce")
        self.X.to_csv("fuck.csv")
        self.bsmt_fin_dict = {None: 0, 'Unf': 1, 'LwQ': 2,
            'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
        self.bsmt_fin_list = ['BsmtFinType1', 'BsmtFinType2']
        for series in self.bsmt_fin_list:
            self.X[series] = pd.to_numeric(self.X[series].map(self.bsmt_fin_dict), errors="coerce")

        self.X['Functional'] = self.X['Functional'].map({None: 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}).astype(int)
        self.X['GarageFinish'] = self.X['GarageFinish'].map({None: 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}).astype(int)
        self.X['Fence'] = self.X['Fence'].map({None: 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}).astype(int)

        self.X['HighSeason'] = self.X['MoSold'].replace( 
        {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})

        self.X['NewerDwelling'] = self.X['MSSubClass'].replace(
        {20: 1, 30: 0, 40: 0, 45: 0,50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
         90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})   
    
        self.X.loc[self.X.Neighborhood == 'NridgHt', 'Neighborhood_Good'] = 1
        self.X.loc[self.X.Neighborhood == 'Crawfor', 'Neighborhood_Good'] = 1
        self.X.loc[self.X.Neighborhood == 'StoneBr', 'Neighborhood_Good'] = 1
        self.X.loc[self.X.Neighborhood == 'Somerst', 'Neighborhood_Good'] = 1
        self.X.loc[self.X.Neighborhood == 'NoRidge', 'Neighborhood_Good'] = 1
        self.X['Neighborhood_Good'].fillna(0, inplace=True)

        self.X['SaleCondition_PriceDown'] = self.X.SaleCondition.replace(
        # {'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0})
        {'Abnorml': 1, 'Alloca': 2, 'AdjLand': 3, 'Family': 4, 'Normal': 5, 'Partial': 0})

        # House completed before sale or not
        self.X['BoughtOffPlan'] = self.X.SaleCondition.replace(
        {'Abnorml' : 0, 'Alloca' : 0, 'AdjLand' : 0, 'Family' : 0, 'Normal' : 0, 'Partial' : 1})
    
        # self.X['BadHeating'] = self.X.HeatingQC.replace(
        #     {'Ex': 0, 'Gd': 0, 'TA': 0, 'Fa': 1, 'Po': 1})
        # self.X['BadHeating'] = self.X.HeatingQC.replace(
        # {'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})

        self.area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea' ]
        self.X['TotalArea'] = self.X[self.area_cols].sum(axis=1)

        self.X['TotalArea1st2nd'] = self.X['1stFlrSF'] + self.X['2ndFlrSF']

        self.X['Age'] = 2010 - self.X['YearBuilt']
        self.X['TimeSinceSold'] = 2010 - self.X['YrSold']

        # If commented - a little bit worse on LB but better in CV
        
    
        self.X['YearsSinceRemodel'] = self.X['YrSold'] - self.X['YearRemodAdd']
        
    def numeric_prep(self, norm=False):
        '''
        Deal with numeric features
            fill NaN with mean
            apply log1p
            normalization(optional)
        '''
        self._numeric_features = [fea for fea in self.X.columns if self.X.dtypes.to_dict()[
            fea] != np.dtype("O")]
        self.X.loc[:, self._numeric_features] = self.X.loc[:, self._numeric_features].apply(
            lambda x: x.fillna(x.mean()))
        self.X.loc[:, self._numeric_features] = self.X.loc[:, self._numeric_features].apply(np.log1p)
        if norm:
            self.X.loc[:, self._numeric_features] = scale(
                self.X.loc[:, self._numeric_features])
        #self.X.to_csv("fuck.csv")

    def vectorizer(self):
        '''
        Vectorizer nun-numeric features to one-hot
        '''
        self.X = pd.get_dummies(self.X)

    def split(self):
        '''
        Split Xtest and Xtrain
        '''
        self.Xtrain = self.X.iloc[:self.train_shape[0], :]
        self.Xtest = self.X.iloc[self.train_shape[0]:, :]

    def apply_all(self, norm=False):
        '''
        Apply all procession
        '''
        self.numeric_prep(norm)
        self.new_features()
       
        self.drop()
        self.vectorizer()
        self.split()
        self.save_csv()

    def save_csv(self):
        """
        save processed data to csv
        """
        self.Xtrain.to_csv("Xtrain.csv", index=False)
        self.Xtest.to_csv("Xtest.csv", index=False)
        self.ytrain.to_csv("ytrain.csv", index=False, header=True)


if __name__ == "__main__":
    fe = FeatureEnge(pd.read_csv("train.csv"), pd.read_csv("test.csv"))
    fe.apply_all()
