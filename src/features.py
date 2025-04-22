# ===== features.py =====
import pandas as pd
import numpy as np

def preprocess(df: pd.DataFrame, is_train: bool = True, fit_columns: list = None):
    # Store IDs and drop
    ids = df['ID']
    df = df.drop('ID', axis=1)

    # --- Temporal Features ---
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce', format='mixed')
    ref_date = df['Dt_Customer'].max() + pd.DateOffset(days=1)
    df['Customer_Tenure'] = (ref_date - df['Dt_Customer']).dt.days
    
    # --- Age Features ---
    current_year = pd.Timestamp.today().year
    df['Age'] = current_year - df['Year_Birth']
    df['Is_Senior'] = (df['Age'] >= 60).astype(int)
    df['Age_Bin'] = pd.cut(df['Age'], 
                         bins=[0, 30, 45, 60, 100], 
                         labels=['Young', 'Adult', 'Middle-aged', 'Senior'])

    # --- Family Structure Features ---
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']
    df['Family_Size'] = df['Total_Children'] + np.where(
        df['Marital_Status'].isin(['Married', 'Together']), 2, 1)
    df['Is_Parent'] = (df['Total_Children'] > 0).astype(int)

    # --- Spending Features ---
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
               'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spent'] = df[mnt_cols].sum(axis=1)
    
    # Spending proportions
    for col in mnt_cols:
        df[f'{col}_Prop'] = df[col] / (df['Total_Spent'] + 1e-5)
    
    # Spending diversity
    df['Spending_Diversity'] = df[mnt_cols].std(axis=1)
    
    # High spender thresholds (based on 75th percentile)
    high_spend_thresholds = {
        'Wines': 600, 'Meat': 800, 'Fish': 150,
        'Sweet': 100, 'Gold': 200, 'Fruits': 50
    }
    df['High_Spender'] = (
        (df['MntWines'] > high_spend_thresholds['Wines']) |
        (df['MntMeatProducts'] > high_spend_thresholds['Meat'])
    ).astype(int)

    # --- Purchase Behavior Features ---
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    df['Web_Vs_Store_Ratio'] = (df['NumWebPurchases'] + 1) / (df['NumStorePurchases'] + 1)
    df['Deal_Usage_Rate'] = df['NumDealsPurchases'] / (df['Total_Purchases'] + 1e-5)
    
    # --- Engagement Features ---
    # df['Response_Rate'] = df['Response'].rolling(window=3, min_periods=1).mean()
    df['Web_Visits_Per_Purchase'] = df['NumWebVisitsMonth'] / (df['NumWebPurchases'] + 1)

    # --- Income Features ---
    df['Income'].fillna(df['Income'].median(), inplace=True)
    df['Income_per_Family_Member'] = df['Income'] / df['Family_Size']
    df['Income_Spent_Ratio'] = df['Total_Spent'] / (df['Income'] + 1e-5)

    # --- Marketing Sensitivity ---
    df['Campaign_Sensitivity'] = df[['AcceptedCmp3', 'AcceptedCmp4', 
                                    'AcceptedCmp5', 'AcceptedCmp1', 
                                    'AcceptedCmp2']].sum(axis=1)
    
    # --- Drop Redundant Columns ---
    drop_cols = ['Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue',
                'Kidhome', 'Teenhome', 'Marital_Status', 'Response']
    df = df.drop(columns=drop_cols, errors='ignore')

    # --- Smart Encoding ---
    df = pd.get_dummies(df, columns=['Education', 'Age_Bin'], 
                       drop_first=True, dummy_na=True)
    
    # Ensure consistent dummy columns between train/test
    if is_train:
        X = df.drop('Target', axis=1)
        y = df['Target']
        return X, y, ids, X.columns.tolist()
    else:
        # Add missing dummy columns present in training
        missing_cols = set(fit_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[fit_columns]
        return df, ids