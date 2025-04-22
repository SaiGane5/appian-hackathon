def preprocess(df: pd.DataFrame, is_train: bool = True):
    # Drop IDs
    ids = df['People ID']
    df = df.drop('People ID', axis=1)

    # Datetime features
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%Y-%m-%d')
    ref = df['Dt_Customer'].max()
    df['Customer_Tenure'] = (ref - df['Dt_Customer']).dt.days

    # Age
    current_year = pd.Timestamp.today().year
    df['Age'] = current_year - df['Year_Birth']
    df = df.drop(['Year_Birth', 'Dt_Customer'], axis=1)

    # Total spend
    mnt_cols = [c for c in df.columns if c.startswith('Mnt')]
    df['Total_Spent'] = df[mnt_cols].sum(axis=1)

    # Total purchases
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)

    # Discount rate
    df['Discount_Rate'] = df['NumDealsPurchases'] / (df['Total_Purchases'] + 1e-5)

    # Encode categoricals
    df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

    # Fill missing incomes
    if df['Income'].isnull().any():
        df['Income'].fillna(df['Income'].median(), inplace=True)

    # Return
    if is_train:
        X = df.drop('Target', axis=1)
        y = df['Target']
        return X, y, ids
    else:
        return df, ids