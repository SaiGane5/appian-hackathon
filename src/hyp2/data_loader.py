import pandas as pd
from datetime import datetime

def load_data(file_path, is_train=True):
    dtype_mapping = {
        'ID': 'int32',
        'Year_Birth': 'int32',
        'Education': 'category',
        'Marital_Status': 'category',
        'Income': 'float32',
        'Kidhome': 'int8',
        'Teenhome': 'int8',
        'Dt_Customer': 'object',
        'Recency': 'int16',
        'Complain': 'int8',
        'MntWines': 'float32',
        'MntFruits': 'float32',
        'MntMeatProducts': 'float32',
        'MntFishProducts': 'float32',
        'MntSweetProducts': 'float32',
        'MntGoldProds': 'float32',
        'NumDealsPurchases': 'int8',
        'AcceptedCmp1': 'int8',
        'AcceptedCmp2': 'int8',
        'AcceptedCmp3': 'int8',
        'AcceptedCmp4': 'int8',
        'AcceptedCmp5': 'int8',
        'NumWebPurchases': 'int8',
        'NumCatalogPurchases': 'int8',
        'NumStorePurchases': 'int8',
        'NumWebVisitsMonth': 'int8'
    }
    if is_train:
        dtype_mapping['Target'] = 'int8'
    df = pd.read_csv(file_path, dtype=dtype_mapping, na_values=['', 'unknown', 'Unknown'])
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    df['Days_Since_Registration'] = (datetime.now() - df['Dt_Customer']).dt.days
    df.drop('Dt_Customer', axis=1, inplace=True)
    df['Income'] = df['Income'].fillna(df['Income'].median())
    for col in ['Education', 'Marital_Status']:
        if 'Unknown' not in df[col].cat.categories:
            df[col] = df[col].cat.add_categories('Unknown')
        df[col] = df[col].fillna('Unknown')
    return df
