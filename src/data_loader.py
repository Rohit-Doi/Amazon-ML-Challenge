import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and basic preprocessing of data."""
    
    def __init__(self, data_dir: str = "dataset"):
        """Initialize the data loader with directory paths.
        
        Args:
            data_dir: Directory containing the dataset files
        """
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "train.csv"
        self.test_path = self.data_dir / "test.csv"
        self.sample_train_path = self.data_dir / "sample_train.csv"
        self.sample_test_path = self.data_dir / "sample_test.csv"
        
    def load_data(self, use_sample: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
        """Load training and test data.
        
        Args:
            use_sample: Whether to use the sample dataset for testing
            
        Returns:
            Tuple containing:
                - X_train: Training features
                - y_train: Training target (None if test data)
                - X_test: Test features
        """
        if use_sample:
            logger.info("Loading sample data...")
            train_path = self.sample_train_path if self.sample_train_path.exists() else self.train_path
            test_path = self.sample_test_path if self.sample_test_path.exists() else self.test_path
        else:
            train_path = self.train_path
            test_path = self.test_path
        
        # Load training data
        logger.info(f"Loading training data from {train_path}...")
        train_df = pd.read_csv(train_path)
        
        # Load test data
        logger.info(f"Loading test data from {test_path}...")
        test_df = pd.read_csv(test_path)
        
        # Separate features and target
        if 'price' in train_df.columns:
            X_train = train_df.drop('price', axis=1)
            y_train = train_df['price']
        else:
            X_train = train_df
            y_train = None
            
        X_test = test_df
        
        logger.info(f"Training data shape: {X_train.shape}")
        if y_train is not None:
            logger.info(f"Target distribution:\n{y_train.describe()}")
        logger.info(f"Test data shape: {X_test.shape}")
        
        return X_train, y_train, X_test

    def get_feature_types(self, df: pd.DataFrame) -> Dict[str, list]:
        """Identify different types of features in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing lists of different feature types
        """
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        text_features = ['catalog_content'] if 'catalog_content' in df.columns else []
        image_features = ['image_link'] if 'image_link' in df.columns else []
        
        # Remove text and image features from categorical features
        categorical_features = [f for f in categorical_features if f not in text_features + image_features]
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'text': text_features,
            'image': image_features
        }
