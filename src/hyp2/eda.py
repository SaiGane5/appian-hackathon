import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_eda(df, target_col='Target'):
    categorical_cols = []
    if 'Education' in df.columns:
        categorical_cols.append('Education')
    if 'Marital_Status' in df.columns:
        categorical_cols.append('Marital_Status')
    print(f"\nData shape: {df.shape}\n")
    print(df.info())
    print("\nSummary statistics:\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())
    
    # Target variable distribution
    if target_col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(x=target_col, data=df)
        plt.title('Target Variable Distribution')
        plt.show()
    # Categorical variables vs Target
    categorical_cols = ['Education', 'Marital_Status']
    for col in categorical_cols:
        if col in df.columns:
            plt.figure(figsize=(12, 6))
            sns.countplot(x=col, hue=target_col, data=df)
            plt.title(f'{col} vs {target_col}')
            plt.xticks(rotation=45)
            plt.show()
    # Numerical variables distributions and boxplots
    numerical_columns = [
        'Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency',
        'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
        'MntSweetProducts', 'MntGoldProds'
    ]
    for col in numerical_columns:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df, x=col, hue=target_col if target_col in df.columns else None, kde=True, element='step', common_norm=False)
            plt.title(f'Distribution of {col} by {target_col}')
            plt.show()
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=target_col, y=col, data=df)
            plt.title(f'{col} by {target_col}')
            plt.show()
    # Correlation heatmap (after encoding categoricals)
    df_encoded = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)
    plt.figure(figsize=(20, 16))
    corr_matrix = df_encoded.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix (Encoded Data)')
    plt.show()
