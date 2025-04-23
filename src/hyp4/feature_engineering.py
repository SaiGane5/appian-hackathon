import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
from sklearn.base import BaseEstimator, TransformerMixin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("appian_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('feature_engineering')

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering pipeline for Appian product purchase prediction.
    Scikit-learn compatible transformer following the fit/transform pattern.
    """
    
    def __init__(self, visualize=True):
        """
        Initialize the feature engineer
        
        Parameters:
        -----------
        visualize : bool
            Whether to generate visualizations during feature engineering
        """
        self.visualize = visualize
        self.debug_dir = 'debug'
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        logger.info("Feature Engineer initialized")
    
    def fit(self, X, y=None):
        """
        Fit the feature engineer to the data (compute feature statistics)
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The input data to fit
        y : pandas.Series, optional
            The target variable
            
        Returns:
        --------
        self : FeatureEngineer
            The fitted feature engineer
        """
        logger.info("Fitting Feature Engineer")
        self.categorical_cols = ['Education', 'Marital_Status']
        
        # Store column names for later use
        self.original_columns = X.columns.tolist()
        
        # Store target column if provided
        self.target_col = y
        
        return self
    
    def transform(self, X):
        """
        Transform the data with feature engineering
        
        Parameters:
        -----------
        X : pandas.DataFrame
            The input data to transform
            
        Returns:
        --------
        pandas.DataFrame
            The transformed data with engineered features
        """
        logger.info("Transforming data with Feature Engineer")
        logger.info(f"Input data shape: {X.shape}")
        
        try:
            # Create a copy to avoid modifying the original
            X_transformed = X.copy()
            
            # Store the ID column for later reference
            id_column = None
            if 'ID' in X_transformed.columns:
                id_column = X_transformed['ID'].copy()
            
            # Debug: Save initial state
            X_transformed.head(5).to_csv(f"{self.debug_dir}/pre_transform_sample.csv")
            
            # === Demographic Features ===
            logger.info("Engineering demographic features")
            
            # Calculate age from birth year
            X_transformed['Age'] = datetime.now().year - X_transformed['Year_Birth']
            
            # Calculate family size
            X_transformed['FamilySize'] = 1 + X_transformed['Kidhome'] + X_transformed['Teenhome']
            
            # Calculate child to adult ratio
            total_children = X_transformed['Kidhome'] + X_transformed['Teenhome']
            X_transformed['ChildAdultRatio'] = total_children / (X_transformed['FamilySize'] - total_children)
            X_transformed['ChildAdultRatio'] = X_transformed['ChildAdultRatio'].replace([np.inf, -np.inf], 0)
            
            # Income per family member
            X_transformed['IncomePerMember'] = X_transformed['Income'] / X_transformed['FamilySize']
            
            # Has children indicator
            X_transformed['HasChildren'] = (total_children > 0).astype(int)
            
            # === Purchase Behavior Features ===
            logger.info("Engineering purchase behavior features")
            
            # Total spending across all categories
            product_cols = [col for col in X_transformed.columns if col.startswith('Mnt')]
            X_transformed['TotalSpent'] = X_transformed[product_cols].sum(axis=1)
            
            # Spending per product category as a percentage of total spending
            for col in product_cols:
                category = col.replace('Mnt', '')
                X_transformed[f'{category}Ratio'] = X_transformed[col] / X_transformed['TotalSpent'].replace(0, 1)
            
            # Spending per family member
            X_transformed['SpentPerPerson'] = X_transformed['TotalSpent'] / X_transformed['FamilySize']
            
            # Volatility in spending (standard deviation across categories)
            X_transformed['SpendingVolatility'] = X_transformed[product_cols].std(axis=1)
            
            # === Purchase Channel Features ===
            logger.info("Engineering purchase channel features")
            
            # Total purchases across all channels
            channel_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
            X_transformed['TotalPurchases'] = X_transformed[channel_cols].sum(axis=1)
            
            # Channel preference ratios
            for col in channel_cols:
                channel = col.replace('Num', '').replace('Purchases', '')
                X_transformed[f'{channel}Ratio'] = X_transformed[col] / X_transformed['TotalPurchases'].replace(0, 1)
            
            # Average basket size (total spent / total purchases)
            X_transformed['AvgBasketSize'] = X_transformed['TotalSpent'] / X_transformed['TotalPurchases'].replace(0, 1)
            
            # Web engagement metric
            X_transformed['WebEngagementRatio'] = X_transformed['NumWebVisitsMonth'] / X_transformed['NumWebPurchases'].replace(0, 1)
            
            # === Campaign Features ===
            logger.info("Engineering campaign features")
            
            # Total accepted campaigns
            campaign_cols = [col for col in X_transformed.columns if col.startswith('AcceptedCmp')]
            X_transformed['TotalAcceptedCampaigns'] = X_transformed[campaign_cols].sum(axis=1)
            
            # Campaign acceptance rate
            X_transformed['CampaignAcceptanceRate'] = X_transformed['TotalAcceptedCampaigns'] / len(campaign_cols)
            
            # === Customer Loyalty & Engagement Features ===
            logger.info("Engineering loyalty and engagement features")
            
            # Customer tenure in years
            X_transformed['TenureYears'] = X_transformed['Days_Since_Registration'] / 365.25
            
            # Purchase frequency (purchases per year)
            X_transformed['PurchaseFrequency'] = X_transformed['TotalPurchases'] / X_transformed['TenureYears'].replace(0, 1)
            
            # Deal seeking behavior
            X_transformed['DealRatio'] = X_transformed['NumDealsPurchases'] / X_transformed['TotalPurchases'].replace(0, 1)
            
            # Recency ratio (recency as percentage of total tenure)
            X_transformed['RecencyRatio'] = X_transformed['Recency'] / X_transformed['Days_Since_Registration'].replace(0, 1)
            
            # === One-hot encoding categorical features ===
            logger.info("One-hot encoding categorical features")
            for col in self.categorical_cols:
                if col in X_transformed.columns:
                    logger.info(f"One-hot encoding {col}")
                    try:
                        X_transformed = pd.get_dummies(X_transformed, columns=[col], prefix=col, drop_first=True)
                    except Exception as e:
                        logger.error(f"Error one-hot encoding {col}: {e}")
                        # Add dummy column if encoding fails
                        X_transformed[f"{col}_Unknown"] = 0
            
            # === Interaction Features ===
            logger.info("Creating interaction features")
            
            # Interaction: Age x Income
            X_transformed['Age_x_Income'] = X_transformed['Age'] * X_transformed['Income'] / 1000  # Scale down
            
            # Interaction: Web engagement x Age
            X_transformed['WebEngagement_x_Age'] = X_transformed['WebEngagementRatio'] * X_transformed['Age'] / 100  # Scale down
            
            # === Feature Visualization ===
            if self.visualize and 'Target' in X_transformed.columns:
                logger.info("Generating feature importance visualizations")
                
                # Select top numerical features to visualize
                num_features = ['Age', 'Income', 'TotalSpent', 'SpentPerPerson', 
                               'CampaignAcceptanceRate', 'TenureYears']
                
                # Create boxplots for each feature by target
                plt.figure(figsize=(15, 10))
                for i, feature in enumerate(num_features, 1):
                    if feature in X_transformed.columns:
                        plt.subplot(2, 3, i)
                        sns.boxplot(x='Target', y=feature, data=X_transformed)
                        plt.title(f"{feature} by Target")
                
                plt.tight_layout()
                plt.savefig(f"{self.debug_dir}/feature_boxplots.png")
                plt.close()
                
                # Plot correlation heatmap of new features
                new_features = ['Age', 'FamilySize', 'TotalSpent', 'SpentPerPerson',
                              'CampaignAcceptanceRate', 'TenureYears', 'PurchaseFrequency']
                new_features = [f for f in new_features if f in X_transformed.columns]
                
                if len(new_features) > 0:
                    plt.figure(figsize=(12, 10))
                    corr_matrix = X_transformed[new_features].corr()
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
                    plt.title('Correlation of Engineered Features')
                    plt.savefig(f"{self.debug_dir}/engineered_features_correlation.png")
                    plt.close()
                
            
            # === Drop Original Features ===
            logger.info("Dropping redundant features")
            
            # List of columns to drop
            to_drop = ['Year_Birth']  # Replaced by Age
            
            # Drop the features
            for col in to_drop:
                if col in X_transformed.columns:
                    X_transformed = X_transformed.drop(col, axis=1)
            
            # Debug: Save post-transformation sample
            X_transformed.head(5).to_csv(f"{self.debug_dir}/post_transform_sample.csv")
            
            # Restore ID column if it was present
            if id_column is not None:
                X_transformed['ID'] = id_column
            
            logger.info(f"Output data shape: {X_transformed.shape}")
            logger.info(f"Engineered features: {set(X_transformed.columns) - set(self.original_columns)}")
            
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}", exc_info=True)
            raise
