import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from scipy import stats
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("appian_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('eda')

class ExploratoryAnalysis:
    """
    Comprehensive exploratory data analysis for Appian purchase prediction.
    Analyzes the raw dataset and produces visualizations to understand patterns.
    """
    
    def __init__(self):
        """Initialize EDA component"""
        self.debug_dir = 'debug'
        self.plots_dir = os.path.join(self.debug_dir, 'plots')
        
        # Create directories if they don't exist
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        
        logger.info("EDA component initialized")
        
        # Define column groups from README.md
        self.people_cols = ['Year_Birth', 'Education', 'Marital_Status', 'Income', 
                           'Kidhome', 'Teenhome', 'Days_Since_Registration', 'Recency', 'Complain']
        
        self.product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                            'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        
        self.promotion_cols = ['NumDealsPurchases', 'AcceptedCmp1', 'AcceptedCmp2', 
                              'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
        
        self.place_cols = ['NumWebPurchases', 'NumCatalogPurchases', 
                          'NumStorePurchases', 'NumWebVisitsMonth']
        
        # Define categorical and numerical columns
        self.categorical_cols = ['Education', 'Marital_Status']
        self.binary_cols = ['Complain', 'AcceptedCmp1', 'AcceptedCmp2', 
                           'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
    
    def run_eda(self, df, target_col='Target'):
        """
        Run comprehensive exploratory data analysis
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The dataset to analyze
        target_col : str
            The name of the target column
        """
        logger.info(f"Running EDA on dataset with shape {df.shape}")
        
        # Store dataset stats
        stats_file = os.path.join(self.debug_dir, 'data_stats.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Dataset Shape: {df.shape}\n\n")
            f.write("Data Types:\n")
            f.write(str(df.dtypes) + "\n\n")
            f.write("Summary Statistics:\n")
            f.write(str(df.describe()) + "\n\n")
            f.write("Missing Values:\n")
            f.write(str(df.isnull().sum()) + "\n\n")
        
        logger.info(f"Basic stats written to {stats_file}")
        
        # Check if target column exists
        has_target = target_col in df.columns
        
        # Run all analysis components
        self._analyze_target_distribution(df, target_col) if has_target else None
        self._analyze_customer_demographics(df, target_col) if has_target else self._analyze_customer_demographics(df)
        self._analyze_purchase_behavior(df, target_col) if has_target else self._analyze_purchase_behavior(df)
        self._analyze_campaign_response(df, target_col) if has_target else self._analyze_campaign_response(df)
        self._analyze_purchase_channels(df, target_col) if has_target else self._analyze_purchase_channels(df)
        self._analyze_correlations(df)
        
        if has_target:
            self._feature_importance_analysis(df, target_col)
        
        logger.info("EDA completed successfully")
        
    def _analyze_target_distribution(self, df, target_col):
        """Analyze target variable distribution"""
        logger.info("Analyzing target distribution")
        
        # Check if target exists
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found in data")
            return
        
        # Calculate target metrics
        target_counts = df[target_col].value_counts()
        class_ratios = target_counts / len(df)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x=target_col, data=df, palette='viridis')
        
        # Add count labels
        for i, count in enumerate(target_counts):
            ax.text(i, count + 5, f"{count} ({class_ratios[i]:.1%})", 
                   ha='center', fontweight='bold')
        
        plt.title('Target Distribution: Will Customer Purchase Appian Product?')
        plt.xlabel('Purchase (1 = Yes, 0 = No)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.plots_dir, 'target_distribution.png'))
        plt.close()
        
        # Log target stats
        logger.info(f"Target distribution: {dict(target_counts)}")
        logger.info(f"Class ratios: {dict(class_ratios)}")
        
        # Save detailed target stats
        stats_path = os.path.join(self.debug_dir, 'target_stats.csv')
        pd.DataFrame({
            'Class': target_counts.index,
            'Count': target_counts.values,
            'Percentage': class_ratios.values * 100
        }).to_csv(stats_path, index=False)
        
    def _analyze_customer_demographics(self, df, target_col=None):
        """Analyze customer demographic data"""
        logger.info("Analyzing customer demographics")
        
        # Create Age from Year_Birth
        if 'Year_Birth' in df.columns:
            current_year = datetime(2025, 4, 23).year  # Using the date from the context
            df['Age'] = current_year - df['Year_Birth']
        
        # Age distribution
        if 'Age' in df.columns:
            plt.figure(figsize=(12, 6))
            
            if target_col in df.columns:
                # Age distribution by target
                sns.histplot(data=df, x='Age', hue=target_col, bins=20, 
                           kde=True, element='step', palette='viridis')
                plt.title('Age Distribution by Purchase Behavior')
            else:
                # Overall age distribution
                sns.histplot(data=df, x='Age', bins=20, kde=True, color='purple')
                plt.title('Customer Age Distribution')
                
            plt.xlabel('Age (Years)')
            plt.ylabel('Count')
            plt.savefig(os.path.join(self.plots_dir, 'age_distribution.png'))
            plt.close()
            
            # Age statistics
            age_stats = df['Age'].describe()
            logger.info(f"Age statistics: mean={age_stats['mean']:.1f}, min={age_stats['min']:.1f}, max={age_stats['max']:.1f}")
        
        # Education distribution
        if 'Education' in df.columns:
            plt.figure(figsize=(12, 6))
            
            if target_col in df.columns:
                education_order = df.groupby('Education')[target_col].mean().sort_values(ascending=False).index
                education_pivot = pd.crosstab(df['Education'], df[target_col], normalize='index')
                education_pivot.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
                plt.title('Education Level vs. Purchase Rate')
                plt.xlabel('Education Level')
                plt.ylabel('Proportion')
                plt.legend(title='Purchase', labels=['No', 'Yes'])
            else:
                order = df['Education'].value_counts().index
                sns.countplot(y='Education', data=df, order=order, palette='viridis')
                plt.title('Distribution of Education Levels')
                plt.xlabel('Count')
                plt.ylabel('Education Level')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'education_distribution.png'))
            plt.close()
        
        # Income analysis
        if 'Income' in df.columns:
            plt.figure(figsize=(12, 6))
            
            if target_col in df.columns:
                # Income boxplot by target
                sns.boxplot(x=target_col, y='Income', data=df, palette='viridis')
                plt.title('Income Distribution by Purchase Behavior')
                plt.xlabel('Purchase (1 = Yes, 0 = No)')
                plt.ylabel('Income')
            else:
                # Overall income distribution
                sns.histplot(data=df, x='Income', bins=30, kde=True, color='purple')
                plt.title('Customer Income Distribution')
                plt.xlabel('Income')
                plt.ylabel('Count')
                
            plt.savefig(os.path.join(self.plots_dir, 'income_distribution.png'))
            plt.close()
            
            # Income statistics
            income_stats = df['Income'].describe()
            logger.info(f"Income statistics: mean={income_stats['mean']:.2f}, median={df['Income'].median():.2f}")
        
        # Family size analysis
        if 'Kidhome' in df.columns and 'Teenhome' in df.columns:
            df['FamilySize'] = 1 + df['Kidhome'] + df['Teenhome']
            
            plt.figure(figsize=(10, 6))
            
            if target_col in df.columns:
                # Family size by target
                crosstab = pd.crosstab(df['FamilySize'], df[target_col], normalize='index')
                crosstab.plot(kind='bar', stacked=True, colormap='viridis')
                plt.title('Family Size vs. Purchase Rate')
                plt.xlabel('Family Size')
                plt.ylabel('Proportion')
                plt.legend(title='Purchase', labels=['No', 'Yes'])
            else:
                # Overall family size distribution
                sns.countplot(x='FamilySize', data=df, palette='viridis')
                plt.title('Family Size Distribution')
                plt.xlabel('Family Size')
                plt.ylabel('Count')
                
            plt.savefig(os.path.join(self.plots_dir, 'family_size_distribution.png'))
            plt.close()
        
        # Tenure analysis
        if 'Days_Since_Registration' in df.columns:
            df['YearsSinceRegistration'] = df['Days_Since_Registration'] / 365.25
            
            plt.figure(figsize=(12, 6))
            
            if target_col in df.columns:
                # Tenure by target
                sns.boxplot(x=target_col, y='YearsSinceRegistration', data=df, palette='viridis')
                plt.title('Customer Tenure by Purchase Behavior')
                plt.xlabel('Purchase (1 = Yes, 0 = No)')
                plt.ylabel('Years Since Registration')
            else:
                # Overall tenure distribution
                sns.histplot(data=df, x='YearsSinceRegistration', bins=20, kde=True, color='purple')
                plt.title('Customer Tenure Distribution')
                plt.xlabel('Years Since Registration')
                plt.ylabel('Count')
                
            plt.savefig(os.path.join(self.plots_dir, 'tenure_distribution.png'))
            plt.close()
    
    def _analyze_purchase_behavior(self, df, target_col=None):
        """Analyze customer purchase behavior"""
        logger.info("Analyzing purchase behavior")
        
        # Total spending analysis
        if all(col in df.columns for col in self.product_cols):
            df['TotalSpend'] = df[self.product_cols].sum(axis=1)
            
            plt.figure(figsize=(12, 6))
            
            if target_col in df.columns:
                # Spending by target (log scale for better visualization)
                sns.boxplot(x=target_col, y='TotalSpend', data=df, palette='viridis')
                plt.yscale('log')
                plt.title('Total Spending by Purchase Behavior (Log Scale)')
                plt.xlabel('Purchase (1 = Yes, 0 = No)')
                plt.ylabel('Total Spend (Log Scale)')
            else:
                # Overall spending distribution
                sns.histplot(data=df, x='TotalSpend', bins=30, kde=True, color='purple')
                plt.title('Total Spending Distribution')
                plt.xlabel('Total Spend')
                plt.ylabel('Count')
                
            plt.savefig(os.path.join(self.plots_dir, 'total_spending_distribution.png'))
            plt.close()
            
            # Spending by category
            plt.figure(figsize=(14, 8))
            
            # Melt the data for easier plotting
            melted_df = pd.melt(df, 
                               id_vars=[target_col] if target_col in df.columns else [],
                               value_vars=self.product_cols,
                               var_name='Category', 
                               value_name='Amount')
            
            # Clean category names
            melted_df['Category'] = melted_df['Category'].str.replace('Mnt', '').str.replace('Products', '')
            
            if target_col in df.columns:
                # Spending by category and target
                sns.boxplot(x='Category', y='Amount', hue=target_col, data=melted_df, palette='viridis')
                plt.title('Spending by Product Category and Purchase Behavior')
                plt.yscale('log')
                plt.legend(title='Purchase', labels=['No', 'Yes'])
            else:
                # Overall spending by category
                sns.boxplot(x='Category', y='Amount', data=melted_df, palette='viridis')
                plt.title('Spending by Product Category')
                plt.yscale('log')
                
            plt.xlabel('Product Category')
            plt.ylabel('Amount Spent (Log Scale)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'category_spending.png'))
            plt.close()
            
            # Spending composition (pie chart)
            plt.figure(figsize=(12, 8))
            category_totals = df[self.product_cols].sum()
            category_names = [col.replace('Mnt', '').replace('Products', '') for col in self.product_cols]
            
            plt.pie(category_totals, labels=category_names, autopct='%1.1f%%', 
                   startangle=90, shadow=True, explode=[0.05]*len(category_totals))
            plt.axis('equal')
            plt.title('Spending Composition by Product Category')
            plt.savefig(os.path.join(self.plots_dir, 'spending_composition.png'))
            plt.close()
        
        # Recency analysis
        if 'Recency' in df.columns:
            plt.figure(figsize=(12, 6))
            
            if target_col in df.columns:
                # Recency by target
                sns.boxplot(x=target_col, y='Recency', data=df, palette='viridis')
                plt.title('Days Since Last Purchase by Purchase Behavior')
                plt.xlabel('Purchase (1 = Yes, 0 = No)')
                plt.ylabel('Days Since Last Purchase')
            else:
                # Overall recency distribution
                sns.histplot(data=df, x='Recency', bins=30, kde=True, color='purple')
                plt.title('Days Since Last Purchase Distribution')
                plt.xlabel('Days Since Last Purchase')
                plt.ylabel('Count')
                
            plt.savefig(os.path.join(self.plots_dir, 'recency_distribution.png'))
            plt.close()
        
        # Complaint analysis
        if 'Complain' in df.columns:
            plt.figure(figsize=(10, 6))
            
            if target_col in df.columns:
                # Complaint by target
                complaint_pivot = pd.crosstab(df['Complain'], df[target_col], normalize='index')
                complaint_pivot.plot(kind='bar', stacked=True, colormap='viridis')
                plt.title('Complaint History vs. Purchase Rate')
                plt.xlabel('Has Complained (1 = Yes, 0 = No)')
                plt.ylabel('Proportion')
                plt.legend(title='Purchase', labels=['No', 'Yes'])
                
                # Add percentage annotations
                totals = df.groupby('Complain')[target_col].count()
                for i, total in enumerate(totals):
                    plt.annotate(f'n={total}', xy=(i, 1.02), ha='center')
            else:
                # Overall complaint distribution
                sns.countplot(x='Complain', data=df, palette='viridis')
                plt.title('Customer Complaint Distribution')
                plt.xlabel('Has Complained (1 = Yes, 0 = No)')
                plt.ylabel('Count')
                
                # Add percentage annotations
                total = len(df)
                for i, count in enumerate(df['Complain'].value_counts()):
                    plt.annotate(f'{count} ({count/total:.1%})', xy=(i, count + 5), ha='center')
                
            plt.savefig(os.path.join(self.plots_dir, 'complaint_distribution.png'))
            plt.close()
    
    def _analyze_campaign_response(self, df, target_col=None):
        """Analyze customer response to campaigns"""
        logger.info("Analyzing campaign response patterns")
        
        # Campaign acceptance analysis
        if all(col in df.columns for col in self.promotion_cols):
            # Create total accepted campaigns column
            df['TotalAcceptedCampaigns'] = df[self.binary_cols[1:]].sum(axis=1)
            
            plt.figure(figsize=(10, 6))
            
            if target_col in df.columns:
                # Campaign acceptance by target
                crosstab = pd.crosstab(df['TotalAcceptedCampaigns'], df[target_col], normalize='index')
                crosstab.plot(kind='bar', stacked=True, colormap='viridis')
                plt.title('Number of Accepted Campaigns vs. Purchase Rate')
                plt.xlabel('Number of Accepted Campaigns')
                plt.ylabel('Proportion')
                plt.legend(title='Purchase', labels=['No', 'Yes'])
            else:
                # Overall campaign acceptance distribution
                sns.countplot(x='TotalAcceptedCampaigns', data=df, palette='viridis')
                plt.title('Number of Accepted Campaigns Distribution')
                plt.xlabel('Number of Accepted Campaigns')
                plt.ylabel('Count')
                
            plt.savefig(os.path.join(self.plots_dir, 'campaign_acceptance_distribution.png'))
            plt.close()
            
            # Campaign effectiveness comparison
            plt.figure(figsize=(12, 6))
            campaign_acceptance = df[self.binary_cols[1:]].mean().reset_index()
            campaign_acceptance.columns = ['Campaign', 'Acceptance_Rate']
            campaign_acceptance['Campaign'] = campaign_acceptance['Campaign'].str.replace('AcceptedCmp', 'Campaign ')
            
            # Plot acceptance rates
            sns.barplot(x='Campaign', y='Acceptance_Rate', data=campaign_acceptance, palette='viridis')
            plt.title('Campaign Effectiveness Comparison')
            plt.xlabel('Campaign')
            plt.ylabel('Acceptance Rate')
            plt.ylim(0, campaign_acceptance['Acceptance_Rate'].max() * 1.2)
            
            # Add percentage labels
            for i, rate in enumerate(campaign_acceptance['Acceptance_Rate']):
                plt.text(i, rate + 0.005, f'{rate:.1%}', ha='center')
                
            plt.savefig(os.path.join(self.plots_dir, 'campaign_effectiveness.png'))
            plt.close()
            
            if target_col in df.columns:
                # Campaign effectiveness by purchase
                plt.figure(figsize=(14, 8))
                campaign_cols = self.binary_cols[1:]
                
                # Calculate acceptance rate by target
                acceptance_by_target = df.groupby(target_col)[campaign_cols].mean().T.reset_index()
                acceptance_by_target = pd.melt(acceptance_by_target, 
                                            id_vars='index',
                                            value_vars=[0, 1],
                                            var_name='Purchase',
                                            value_name='Acceptance_Rate')
                
                acceptance_by_target['Campaign'] = acceptance_by_target['index'].str.replace('AcceptedCmp', 'Campaign ')
                acceptance_by_target['Purchase'] = acceptance_by_target['Purchase'].map({0: 'No Purchase', 1: 'Purchase'})
                
                # Plot by target
                sns.barplot(x='Campaign', y='Acceptance_Rate', hue='Purchase', data=acceptance_by_target, palette='viridis')
                plt.title('Campaign Effectiveness by Purchase Behavior')
                plt.xlabel('Campaign')
                plt.ylabel('Acceptance Rate')
                plt.legend(title='Appian Product Purchase')
                plt.savefig(os.path.join(self.plots_dir, 'campaign_by_purchase.png'))
                plt.close()
        
        # Deals analysis
        if 'NumDealsPurchases' in df.columns:
            plt.figure(figsize=(12, 6))
            
            if target_col in df.columns:
                # Deals by target
                sns.boxplot(x=target_col, y='NumDealsPurchases', data=df, palette='viridis')
                plt.title('Number of Deals Purchases by Purchase Behavior')
                plt.xlabel('Purchase (1 = Yes, 0 = No)')
                plt.ylabel('Number of Deals Purchases')
            else:
                # Overall deals distribution
                sns.countplot(x='NumDealsPurchases', data=df, palette='viridis')
                plt.title('Number of Deals Purchases Distribution')
                plt.xlabel('Number of Deals Purchases')
                plt.ylabel('Count')
                
            plt.savefig(os.path.join(self.plots_dir, 'deals_distribution.png'))
            plt.close()
    
    def _analyze_purchase_channels(self, df, target_col=None):
        """Analyze customer purchase channels"""
        logger.info("Analyzing purchase channels")
        
        # Purchase channels analysis
        if all(col in df.columns for col in self.place_cols[:-1]):
            # Calculate total purchases across channels
            df['TotalPurchases'] = df[self.place_cols[:-1]].sum(axis=1)
            
            # Calculate channel proportions
            for channel in self.place_cols[:-1]:
                channel_name = channel.replace('Num', '').replace('Purchases', '')
                df[f'{channel_name}Proportion'] = df[channel] / df['TotalPurchases'].replace(0, 1)
            
            # Channel distribution
            plt.figure(figsize=(12, 8))
            
            # Create data for pie chart
            channel_totals = df[self.place_cols[:-1]].sum()
            channel_names = [col.replace('Num', '').replace('Purchases', '') for col in self.place_cols[:-1]]
            
            plt.pie(channel_totals, labels=channel_names, autopct='%1.1f%%', 
                   startangle=90, shadow=True, explode=[0.05]*len(channel_totals))
            plt.axis('equal')
            plt.title('Purchase Distribution by Channel')
            plt.savefig(os.path.join(self.plots_dir, 'channel_distribution.png'))
            plt.close()
            
            if target_col in df.columns:
                # Channel preference by target
                plt.figure(figsize=(14, 8))
                
                # Calculate average proportion by target
                channel_props = ['WebProportion', 'CatalogProportion', 'StoreProportion']
                channel_by_target = df.groupby(target_col)[channel_props].mean().T.reset_index()
                channel_by_target = pd.melt(channel_by_target,
                                          id_vars='index',
                                          value_vars=[0, 1],
                                          var_name='Purchase',
                                          value_name='Proportion')
                
                channel_by_target['Channel'] = channel_by_target['index'].str.replace('Proportion', '')
                channel_by_target['Purchase'] = channel_by_target['Purchase'].map({0: 'No Purchase', 1: 'Purchase'})
                
                # Plot by target
                sns.barplot(x='Channel', y='Proportion', hue='Purchase', data=channel_by_target, palette='viridis')
                plt.title('Channel Preference by Purchase Behavior')
                plt.xlabel('Channel')
                plt.ylabel('Proportion of Purchases')
                plt.legend(title='Appian Product Purchase')
                plt.savefig(os.path.join(self.plots_dir, 'channel_by_purchase.png'))
                plt.close()
        
        # Web visits analysis
        if 'NumWebVisitsMonth' in df.columns:
            plt.figure(figsize=(12, 6))
            
            if target_col in df.columns:
                # Web visits by target
                sns.boxplot(x=target_col, y='NumWebVisitsMonth', data=df, palette='viridis')
                plt.title('Number of Website Visits by Purchase Behavior')
                plt.xlabel('Purchase (1 = Yes, 0 = No)')
                plt.ylabel('Number of Website Visits Last Month')
            else:
                # Overall web visits distribution
                sns.countplot(x='NumWebVisitsMonth', data=df, palette='viridis')
                plt.title('Number of Website Visits Distribution')
                plt.xlabel('Number of Website Visits Last Month')
                plt.ylabel('Count')
                
            plt.savefig(os.path.join(self.plots_dir, 'web_visits_distribution.png'))
            plt.close()
            
            # Web engagement ratio
            if 'NumWebPurchases' in df.columns:
                df['WebConversionRatio'] = df['NumWebPurchases'] / df['NumWebVisitsMonth'].replace(0, 1)
                
                plt.figure(figsize=(12, 6))
                
                if target_col in df.columns:
                    # Web conversion by target
                    sns.boxplot(x=target_col, y='WebConversionRatio', data=df, palette='viridis')
                    plt.title('Web Conversion Ratio by Purchase Behavior')
                    plt.xlabel('Purchase (1 = Yes, 0 = No)')
                    plt.ylabel('Web Conversion Ratio')
                else:
                    # Overall web conversion distribution
                    sns.histplot(data=df, x='WebConversionRatio', bins=30, kde=True, color='purple')
                    plt.title('Web Conversion Ratio Distribution')
                    plt.xlabel('Web Conversion Ratio')
                    plt.ylabel('Count')
                    
                plt.savefig(os.path.join(self.plots_dir, 'web_conversion_distribution.png'))
                plt.close()
    
    def _analyze_correlations(self, df):
        """Analyze feature correlations"""
        logger.info("Analyzing feature correlations")
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) > 0:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Plot correlation heatmap
            plt.figure(figsize=(20, 16))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'correlation_matrix.png'))
            plt.close()
            
            # Save correlation matrix to CSV
            corr_matrix.to_csv(os.path.join(self.debug_dir, 'correlation_matrix.csv'))
            
            # If target exists, show top correlations with target
            if 'Target' in corr_matrix.columns:
                target_corr = corr_matrix['Target'].sort_values(ascending=False)
                
                # Plot top correlations with target
                plt.figure(figsize=(12, 8))
                top_n = min(15, len(target_corr) - 1)  # Exclude target's correlation with itself
                sns.barplot(x=target_corr.values[1:top_n+1], y=target_corr.index[1:top_n+1], palette='viridis')
                plt.title(f'Top {top_n} Feature Correlations with Target')
                plt.xlabel('Correlation Coefficient')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plots_dir, 'target_correlations.png'))
                plt.close()
                
                # Log top correlations
                logger.info(f"Top 5 positive correlations with target: {target_corr[1:6].to_dict()}")
                logger.info(f"Top 5 negative correlations with target: {target_corr[-5:].to_dict()}")
                
                # Save target correlations to CSV
                target_corr.to_frame().to_csv(os.path.join(self.debug_dir, 'target_correlations.csv'))
    
    def _feature_importance_analysis(self, df, target_col):
        """Analyze feature importance using simple models"""
        try:
            logger.info("Analyzing feature importance")
            
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            # Process data for modeling
            X = df.select_dtypes(include=['number']).drop([target_col], axis=1, errors='ignore')
            
            # Skip if not enough features
            if X.shape[1] < 2:
                logger.warning("Not enough numeric features for importance analysis")
                return
                
            y = df[target_col]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train a simple random forest for feature importance
            rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(X_scaled, y)
            
            # Get feature importance
            importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importances.head(15), palette='viridis')
            plt.title('Top 15 Features by Random Forest Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'feature_importance.png'))
            plt.close()
            
            # Save feature importance to CSV
            importances.to_csv(os.path.join(self.debug_dir, 'feature_importance.csv'), index=False)
            
            logger.info(f"Top 5 important features: {importances.head(5)['Feature'].tolist()}")
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}", exc_info=True)


def run_eda(df, target_col='Target'):
    """
    Run exploratory data analysis on the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    target_col : str
        The name of the target column
    """
    analyzer = ExploratoryAnalysis()
    analyzer.run_eda(df, target_col)
