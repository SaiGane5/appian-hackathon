import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import KNNImputer
from typing import Tuple, List, Dict, Optional

def preprocess(df: pd.DataFrame, is_train: bool = True, 
               fit_columns: Optional[List[str]] = None,
               scalers: Optional[Dict] = None) -> Tuple:
    """
    Enhanced preprocessing pipeline for customer purchase prediction.
    
    Args:
        df: Input dataframe
        is_train: Whether processing training data
        fit_columns: List of column names from training data
        scalers: Dictionary of fitted scalers to apply to test data
        
    Returns:
        Processed data, target (if train), IDs, and column names
    """
    # Store IDs
    ids = df['ID'].copy()
    df = df.drop('ID', axis=1)
    
    # Create copies of original data for tracking changes
    orig_df = df.copy()
    
    # --- Handle Missing Values First ---
    if 'Income' in df.columns:
        df['Income'] = df['Income'].fillna(df['Income'].median())
    
    # --- Temporal Features ---
    df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
    ref_date = pd.to_datetime('2023-01-01')  # Using a fixed reference date for consistency
    df['Customer_Tenure_Days'] = (ref_date - df['Dt_Customer']).dt.days
    df['Customer_Tenure_Months'] = df['Customer_Tenure_Days'] / 30.44
    df['Customer_Tenure_Years'] = df['Customer_Tenure_Days'] / 365.25
    
    # Extract date components
    df['Enroll_Year'] = df['Dt_Customer'].dt.year
    df['Enroll_Month'] = df['Dt_Customer'].dt.month
    df['Enroll_Quarter'] = df['Dt_Customer'].dt.quarter
    df['Enroll_Day'] = df['Dt_Customer'].dt.day
    df['Enroll_DayOfWeek'] = df['Dt_Customer'].dt.dayofweek
    df['Enroll_Weekend'] = (df['Enroll_DayOfWeek'] >= 5).astype(int)
    
    # Recency ratio feature
    df['Recency_Tenure_Ratio'] = df['Recency'] / (df['Customer_Tenure_Days'] + 1)
    
    # --- Age Features ---
    current_year = 2023  # Using a fixed year for consistency
    df['Age'] = current_year - df['Year_Birth']
    
    # Age segments with more granularity
    df['Age_Bin'] = pd.cut(
        df['Age'], 
        bins=[0, 25, 35, 45, 55, 65, 100], 
        labels=['GenZ', 'Millennial', 'GenX_Young', 'GenX_Old', 'Boomer', 'Senior']
    )
    
    # --- Family Structure Features ---
    df['Total_Children'] = df['Kidhome'] + df['Teenhome']
    df['Has_Partner'] = df['Marital_Status'].isin(['Married', 'Together']).astype(int)
    df['Family_Size'] = df['Total_Children'] + np.where(df['Has_Partner'] == 1, 2, 1)
    df['Is_Parent'] = (df['Total_Children'] > 0).astype(int)
    df['Empty_Nester'] = ((df['Age'] > 50) & (df['Total_Children'] == 0) & 
                         (df['Has_Partner'] == 1)).astype(int)
    
    # Family lifecycle stage
    conditions = [
        (df['Age'] < 35) & (df['Total_Children'] == 0) & (df['Has_Partner'] == 0),
        (df['Age'] < 45) & (df['Total_Children'] == 0) & (df['Has_Partner'] == 1),
        (df['Total_Children'] > 0) & (df['Kidhome'] > 0),
        (df['Total_Children'] > 0) & (df['Teenhome'] > 0) & (df['Kidhome'] == 0),
        (df['Age'] > 50) & (df['Total_Children'] == 0)
    ]
    choices = ['Young_Single', 'Young_Couple', 'Parents_Young', 'Parents_Teen', 'Empty_Nest']
    df['Family_Lifecycle'] = pd.Series(np.select(conditions, choices, default='Other'), index=df.index)
    
    # --- Spending Features ---
    mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
               'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    
    # Total spending
    df['Total_Spent'] = df[mnt_cols].sum(axis=1)
    
    # Category preferences
    df['Luxury_Ratio'] = (df['MntWines'] + df['MntGoldProds']) / (df['Total_Spent'] + 1)
    df['Food_Ratio'] = (df['MntFruits'] + df['MntMeatProducts'] + 
                        df['MntFishProducts'] + df['MntSweetProducts']) / (df['Total_Spent'] + 1)
    
    # Spending ratios
    for col in mnt_cols:
        df[f'{col}_Ratio'] = df[col] / (df['Total_Spent'] + 1)
    
    # Spending per capita
    df['Spent_Per_Person'] = df['Total_Spent'] / df['Family_Size']
    
    # Spending diversity (coefficient of variation)
    df['Spending_CV'] = df[mnt_cols].std(axis=1) / (df[mnt_cols].mean(axis=1) + 1)
    
    # Product combinations
    df['Wine_And_Meat'] = ((df['MntWines'] > df['MntWines'].median()) & 
                          (df['MntMeatProducts'] > df['MntMeatProducts'].median())).astype(int)
    df['Sweet_And_Gold'] = ((df['MntSweetProducts'] > df['MntSweetProducts'].median()) & 
                           (df['MntGoldProds'] > df['MntGoldProds'].median())).astype(int)
    
    # --- Purchase Channel Behavior ---
    purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis=1)
    
    # Channel preferences
    for col in purchase_cols:
        df[f'{col}_Ratio'] = df[col] / (df['Total_Purchases'] + 1)
    
    df['Web_Store_Ratio'] = (df['NumWebPurchases'] + 1) / (df['NumStorePurchases'] + 1)
    df['Catalog_Store_Ratio'] = (df['NumCatalogPurchases'] + 1) / (df['NumStorePurchases'] + 1)
    
    # Purchase efficiency
    df['Avg_Spent_Per_Purchase'] = df['Total_Spent'] / (df['Total_Purchases'] + 1)
    df['Deal_Usage_Rate'] = df['NumDealsPurchases'] / (df['Total_Purchases'] + 1)
    
    # Web engagement
    df['Web_Visits_Per_Purchase'] = df['NumWebVisitsMonth'] / (df['NumWebPurchases'] + 1)
    df['Web_Conversion_Rate'] = df['NumWebPurchases'] / (df['NumWebVisitsMonth'] + 1)
    
    # --- Income Features ---
    df['Income_Per_Family_Member'] = df['Income'] / df['Family_Size']
    df['Spending_Income_Ratio'] = df['Total_Spent'] / (df['Income'] + 1)
    df['Luxury_Income_Ratio'] = (df['MntWines'] + df['MntGoldProds']) / (df['Income'] + 1)
    
    # Income quantiles for advanced segmentation
    df['Income_Quantile'] = pd.qcut(df['Income'], q=5, labels=False)
    
    # --- Marketing Sensitivity ---
    campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
                    'AcceptedCmp4', 'AcceptedCmp5']
    df['Campaign_Accepted_Count'] = df[campaign_cols].sum(axis=1)
    df['Campaign_Response_Rate'] = df['Campaign_Accepted_Count'] / len(campaign_cols)
    
    # Recent campaign success
    df['Recent_Campaign_Success'] = df[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']].sum(axis=1)
    df['Early_Campaign_Success'] = df[['AcceptedCmp1', 'AcceptedCmp2']].sum(axis=1)
    
    # Customer feedback metric
    df['Customer_Satisfaction'] = 1 - df['Complain']
    
    # --- Interaction Features ---
    # Income + Age interactions
    df['Income_X_Age'] = df['Income'] * df['Age']
    
    # Recency + Spending interactions
    df['Recency_X_Spending'] = df['Recency'] * df['Total_Spent']
    df['Recent_High_Spender'] = ((df['Recency'] < df['Recency'].median()) & 
                                (df['Total_Spent'] > df['Total_Spent'].median())).astype(int)
    
    # Campaign response + spending
    df['Campaign_Success_X_Spending'] = df['Campaign_Accepted_Count'] * df['Total_Spent']
    
    # --- RFM Analysis Features ---
    # We already have Recency
    # Frequency = Total Purchases
    # Monetary = Total Spent
    
    # RFM Combined Score (normalized)
    df['R_Score'] = pd.qcut(df['Recency'], q=5, labels=False, duplicates='drop')
    df['R_Score'] = 4 - df['R_Score']  # Inverse (lower recency is better)
    df['F_Score'] = pd.qcut(df['Total_Purchases'].clip(lower=1), q=5, labels=False, duplicates='drop')
    df['M_Score'] = pd.qcut(df['Total_Spent'].clip(lower=1), q=5, labels=False, duplicates='drop')
    df['RFM_Score'] = df['R_Score'] + df['F_Score'] + df['M_Score']
    
    # --- Customer Segmentation ---
    # High/Med/Low value customers
    df['Customer_Value'] = pd.qcut(df['Total_Spent'], q=3, labels=['Low', 'Medium', 'High'])
    
    # Active vs Inactive
    df['Is_Active'] = (df['Recency'] < 90).astype(int)
    
    # --- Education Processing ---
    # Education level ordering
    education_order = {
        'Basic': 0,
        '2n Cycle': 1,
        'Graduation': 2,
        'Master': 3,
        'PhD': 4
    }
    
    # Convert education to ordinal if present
    if 'Education' in df.columns:
        df['Education_Ordinal'] = df['Education'].map(education_order).fillna(-1).astype(int)
    
    # --- Drop Redundant Columns ---
    cols_to_drop = ['Year_Birth', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
    
    # --- Encoding ---
    # One-hot encode categorical variables
    cat_cols = ['Marital_Status', 'Education', 'Age_Bin', 'Family_Lifecycle', 'Customer_Value']
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dummy_na=True)
    
    # --- Feature Scaling ---
    # These columns benefit from scaling/transformation
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'Target' in numeric_cols and is_train:
        numeric_cols.remove('Target')
    
    if is_train:
        # Initialize scalers
        scalers = {
            'standard': StandardScaler(),
            'power': PowerTransformer(method='yeo-johnson')
        }
        
        # Columns for standard scaling
        std_cols = [col for col in numeric_cols if 'Ratio' in col or 'Rate' in col]
        # Columns for power transformation (handle skewed distributions)
        power_cols = ['Income', 'Total_Spent', 'Spent_Per_Person', 'Income_Per_Family_Member']
        power_cols = [col for col in power_cols if col in numeric_cols]
        
        # Fit and transform
        if std_cols:
            df_std = pd.DataFrame(
                scalers['standard'].fit_transform(df[std_cols]),
                columns=[f'{col}_scaled' for col in std_cols],
                index=df.index
            )
            df = pd.concat([df, df_std], axis=1)
            
        if power_cols:
            df_power = pd.DataFrame(
                scalers['power'].fit_transform(df[power_cols].fillna(0)),
                columns=[f'{col}_norm' for col in power_cols],
                index=df.index
            )
            df = pd.concat([df, df_power], axis=1)
    else:
        # Apply pre-fitted scalers to test data
        if scalers and 'standard' in scalers:
            std_cols = [col for col in numeric_cols if 'Ratio' in col or 'Rate' in col]
            if std_cols:
                df_std = pd.DataFrame(
                    scalers['standard'].transform(df[std_cols]),
                    columns=[f'{col}_scaled' for col in std_cols],
                    index=df.index
                )
                df = pd.concat([df, df_std], axis=1)
                
        if scalers and 'power' in scalers:
            power_cols = ['Income', 'Total_Spent', 'Spent_Per_Person', 'Income_Per_Family_Member']
            power_cols = [col for col in power_cols if col in numeric_cols]
            if power_cols:
                df_power = pd.DataFrame(
                    scalers['power'].transform(df[power_cols].fillna(0)),
                    columns=[f'{col}_norm' for col in power_cols],
                    index=df.index
                )
                df = pd.concat([df, df_power], axis=1)
    
    # Return processed data
    if is_train:
        if 'Target' in df.columns:
            X = df.drop('Target', axis=1) 
            y = df['Target']
        else:
            raise ValueError("Target column not found in training data")
        return X, y, ids, X.columns.tolist(), scalers
    else:
        # Add missing columns to test data
        if fit_columns:
            missing_cols = set(fit_columns) - set(df.columns)
            for col in missing_cols:
                df[col] = 0
            # Ensure column order matches training data
            df = df[fit_columns]
        return df, ids