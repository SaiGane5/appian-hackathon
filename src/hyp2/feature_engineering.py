import pandas as pd

class FeatureEngineerer:
    def __init__(self):
        self.categorical_cols = ['Education', 'Marital_Status']
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        # Age
        current_year = pd.Timestamp.now().year
        X['Age'] = current_year - X['Year_Birth']
        # Family size
        X['FamilySize'] = 1 + X['Kidhome'] + X['Teenhome']
        # Total spent
        spent_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        X['TotalSpent'] = X[spent_cols].sum(axis=1)
        X['SpentPerPerson'] = X['TotalSpent'] / X['FamilySize']
        # Purchase channel ratios
        channels = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        X['TotalPurchases'] = X[channels].sum(axis=1)
        for ch in channels:
            X[f'{ch}_Ratio'] = X[ch] / X['TotalPurchases'].replace(0, 1)
        # Campaign acceptance rate
        campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
        X['CampaignAcceptanceRate'] = X[campaign_cols].sum(axis=1) / len(campaign_cols)
        # One-hot encode categoricals
        df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)
        df = df.drop(['Education', 'Marital_Status'], axis=1)
        # Drop unused columns
        X = X.drop(['ID', 'Year_Birth'], axis=1, errors='ignore')
        return X
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
