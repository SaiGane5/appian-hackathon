import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("appian_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('predict')

class ModelPredictor:
    """
    Model prediction component for Appian purchase prediction.
    Makes predictions on new data using trained models.
    """
    
    def __init__(self, model_dir='models', output_dir='submissions'):
        """
        Initialize model predictor
        
        Parameters:
        -----------
        model_dir : str
            Directory where models are stored
        output_dir : str
            Directory where to save prediction outputs
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.debug_dir = 'debug'
        self.plots_dir = os.path.join(self.debug_dir, 'prediction_plots')
        
        # Create directories if they don't exist
        for directory in [self.output_dir, self.debug_dir, self.plots_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        self.model = None
        self.feature_columns = None
        
        logger.info("ModelPredictor initialized")
    
    def load_model(self, model_path='rf_model.pkl', features_path=None):
        """
        Load the trained model and feature columns
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        features_path : str
            Path to the saved feature columns file
        """
        try:
            # Resolve model path
            if not os.path.isabs(model_path):
                model_path = os.path.join(self.model_dir, model_path)
            
            # Load model
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            
            # Determine features path if not provided
            if features_path is None:
                features_path = os.path.splitext(model_path)[0] + '_features.json'
                if not os.path.exists(features_path):
                    features_path = os.path.join(self.model_dir, 'feature_columns.json')
            
            # Load feature columns
            if os.path.exists(features_path):
                logger.info(f"Loading feature columns from {features_path}")
                with open(features_path, 'r') as f:
                    self.feature_columns = json.load(f)
            else:
                logger.warning(f"Feature columns file not found at {features_path}")
                self.feature_columns = None
            
            logger.info(f"Model loaded successfully: {type(self.model).__name__}")
            
            # Log model details if possible
            if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
                classifier = self.model.named_steps['classifier']
                classifier_name = type(classifier).__name__
                logger.info(f"Classifier type: {classifier_name}")
                
                # Log additional details based on model type
                if hasattr(classifier, 'n_estimators'):
                    logger.info(f"Number of estimators: {classifier.n_estimators}")
                if hasattr(classifier, 'max_depth'):
                    logger.info(f"Max depth: {classifier.max_depth}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return False
    
    def prepare_features(self, test_df):
        """
        Prepare features for prediction
        
        Parameters:
        -----------
        test_df : pandas.DataFrame
            Test data
        
        Returns:
        --------
        pandas.DataFrame
            Prepared features
        """
        try:
            logger.info("Preparing features for prediction")
            
            # Save a copy of the ID column
            id_column = test_df['ID'].copy() if 'ID' in test_df.columns else None
            
            # Check if feature columns are available
            if self.feature_columns is not None:
                logger.info(f"Aligning features with training schema ({len(self.feature_columns)} features)")
                
                # Find missing columns
                missing_cols = set(self.feature_columns) - set(test_df.columns)
                if missing_cols:
                    logger.warning(f"Missing columns in test data: {missing_cols}")
                    # Add missing columns with zeros
                    for col in missing_cols:
                        test_df[col] = 0
                
                # Select only the features used during training
                test_features = test_df[self.feature_columns].copy()
                
                # Check if any columns have NaN values
                na_counts = test_features.isna().sum()
                has_na = na_counts.sum() > 0
                if has_na:
                    logger.warning(f"Test data contains missing values:\n{na_counts[na_counts > 0]}")
                    # Fill missing values with zeros
                    test_features = test_features.fillna(0)
                
                # Debug: Save sample of prepared features
                test_features.head(5).to_csv(os.path.join(self.debug_dir, 'test_features_sample.csv'))
                
                logger.info(f"Features prepared: shape={test_features.shape}")
                return test_features, id_column
            else:
                logger.warning("No feature columns available, using all columns in test data")
                
                # Save a copy of the ID column if it exists
                id_column = test_df['ID'].copy() if 'ID' in test_df.columns else None
                
                # Remove ID column from features
                if 'ID' in test_df.columns:
                    test_features = test_df.drop('ID', axis=1).copy()
                else:
                    test_features = test_df.copy()
                
                # Debug: Save sample of prepared features
                test_features.head(5).to_csv(os.path.join(self.debug_dir, 'test_features_sample_no_schema.csv'))
                
                logger.info(f"Features prepared (no schema): shape={test_features.shape}")
                return test_features, id_column
                
        except Exception as e:
            logger.error(f"Error preparing features: {e}", exc_info=True)
            raise
    
    def make_predictions(self, test_features):
        """
        Make predictions on test features
        
        Parameters:
        -----------
        test_features : pandas.DataFrame
            Prepared test features
        
        Returns:
        --------
        tuple
            (predicted_labels, predicted_probabilities)
        """
        try:
            logger.info("Making predictions")
            
            if self.model is None:
                logger.error("Model not loaded")
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Make predictions
            y_pred = self.model.predict(test_features)
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(test_features)[:, 1]
                
                # Log prediction stats
                logger.info(f"Predictions made: {len(y_pred)}")
                logger.info(f"Positive predictions: {np.sum(y_pred == 1)} ({np.mean(y_pred == 1):.2%})")
                logger.info(f"Probability stats: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")
                
                # Plot probability distribution
                plt.figure(figsize=(10, 6))
                sns.histplot(y_pred_proba, bins=30, kde=True)
                plt.title('Prediction Probability Distribution')
                plt.xlabel('Probability of Purchase')
                plt.ylabel('Count')
                plt.savefig(os.path.join(self.plots_dir, 'prediction_probabilities.png'))
                plt.close()
                
                return y_pred, y_pred_proba
            else:
                logger.info(f"Predictions made: {len(y_pred)}")
                logger.info(f"Positive predictions: {np.sum(y_pred == 1)} ({np.mean(y_pred == 1):.2%})")
                return y_pred, None
                
        except Exception as e:
            logger.error(f"Error making predictions: {e}", exc_info=True)
            raise
    
    def create_submission(self, id_column, predictions, probabilities=None, 
                         filename=None, include_probabilities=False):
        """
        Create submission file
        
        Parameters:
        -----------
        id_column : pandas.Series
            ID column from test data
        predictions : numpy.ndarray
            Predicted labels
        probabilities : numpy.ndarray, optional
            Predicted probabilities
        filename : str, optional
            Output filename
        include_probabilities : bool
            Whether to include probabilities in output
        
        Returns:
        --------
        pandas.DataFrame
            Submission dataframe
        """
        try:
            logger.info("Creating submission file")
            
            # Create timestamp for filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"submission_{timestamp}.csv"
            
            # Ensure the filename has .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Create full path
            output_path = os.path.join(self.output_dir, filename)
            
            # Create submission dataframe
            submission = pd.DataFrame({'ID': id_column, 'Target': predictions.astype(int)})
            
            # Add probabilities if available and requested
            if include_probabilities and probabilities is not None:
                submission['Probability'] = probabilities
            
            # Save submission file
            submission.to_csv(output_path, index=False)
            logger.info(f"Submission saved to {output_path}")
            
            # Also save a copy to debug directory for inspection
            debug_path = os.path.join(self.debug_dir, f"debug_{filename}")
            submission.to_csv(debug_path, index=False)
            
            # Log a sample of the submission
            logger.info(f"Submission sample:\n{submission.head()}")
            
            return submission
            
        except Exception as e:
            logger.error(f"Error creating submission: {e}", exc_info=True)
            raise
    
    def predict(self, test_df, model_path='rf_model.pkl', 
               submission_filename=None, include_probabilities=False):
        """
        End-to-end prediction pipeline
        
        Parameters:
        -----------
        test_df : pandas.DataFrame
            Test data
        model_path : str
            Path to the model file
        submission_filename : str, optional
            Output filename
        include_probabilities : bool
            Whether to include probabilities in output
        
        Returns:
        --------
        pandas.DataFrame
            Submission dataframe
        """
        try:
            logger.info("Starting prediction pipeline")
            
            # Load model
            success = self.load_model(model_path)
            if not success:
                raise ValueError(f"Failed to load model from {model_path}")
            
            # Prepare features
            test_features, id_column = self.prepare_features(test_df)
            
            # Make predictions
            predictions, probabilities = self.make_predictions(test_features)
            
            # Create submission
            submission = self.create_submission(
                id_column, 
                predictions, 
                probabilities,
                submission_filename,
                include_probabilities
            )
            
            logger.info("Prediction pipeline completed successfully")
            return submission
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}", exc_info=True)
            raise


def predict_and_save(test_df, model_path='rf_model.pkl', output_file='submission.csv'):
    """
    Make predictions on test data and save to a submission file
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        Test data with features
    model_path : str
        Path to the trained model
    output_file : str
        Path to save submission file
    
    Returns:
    --------
    pandas.DataFrame
        Submission dataframe
    """
    try:
        logger.info(f"Starting prediction with model {model_path}")
        
        # Initialize predictor
        predictor = ModelPredictor()
        
        # Make predictions
        submission = predictor.predict(test_df, model_path, output_file)
        
        logger.info(f"Predictions saved to {output_file}")
        return submission
        
    except Exception as e:
        logger.error(f"Error in predict_and_save: {e}", exc_info=True)
        raise
