import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from category_encoders import TargetEncoder
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import cv2
import os

logger = logging.getLogger(__name__)

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from text data using TF-IDF and optionally BERT embeddings."""
    
    def __init__(self, text_columns: List[str], use_bert: bool = False):
        self.text_columns = text_columns
        self.use_bert = use_bert
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.bert_model = None
        self.bert_tokenizer = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """Fit the text feature extractor."""
        if len(self.text_columns) == 0:
            return self
            
        # Fit TF-IDF
        text_data = X[self.text_columns[0]].fillna('')
        self.tfidf.fit(text_data)
        
        # Optionally load BERT model
        if self.use_bert:
            logger.info("Loading BERT model for text embeddings...")
            self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
            
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform text data into features."""
        if len(self.text_columns) == 0:
            return np.array([]).reshape(len(X), 0)
            
        text_data = X[self.text_columns[0]].fillna('')
        
        # Get TF-IDF features
        tfidf_features = self.tfidf.transform(text_data)
        tfidf_features = self.svd.fit_transform(tfidf_features)
        
        if not self.use_bert:
            return tfidf_features
            
        # Get BERT embeddings (mean pooling)
        with torch.no_grad():
            inputs = self.bert_tokenizer(
                text_data.tolist(), 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            )
            outputs = self.bert_model(**inputs)
            # Mean pooling
            attention_mask = inputs['attention_mask']
            last_hidden = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            bert_embeddings = sum_embeddings / sum_mask
            bert_embeddings = bert_embeddings.numpy()
        
        # Combine TF-IDF and BERT features
        return np.hstack([tfidf_features, bert_embeddings])

class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features from image URLs using a pre-trained CNN."""
    
    def __init__(self, image_column: str, use_pretrained: bool = True):
        self.image_column = image_column
        self.use_pretrained = use_pretrained
        self.model = None
        self.preprocess = None
        
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image."""
        try:
            # Load image from URL or local path
            if image_path.startswith(('http://', 'https://')):
                # For demo purposes - in practice, you'd download the image
                # and process it properly
                img = np.random.rand(224, 224, 3)  # Placeholder
            else:
                if not os.path.exists(image_path):
                    return np.zeros((224, 224, 3))  # Return black image if not found
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize and normalize
            img = cv2.resize(img, (224, 224))
            return img.astype('float32') / 255.0
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {str(e)}")
            return np.zeros((224, 224, 3))
    
    def fit(self, X: pd.DataFrame, y=None):
        """Load the pre-trained model if needed."""
        if not self.image_column or self.image_column not in X.columns:
            return self
            
        if self.use_pretrained and self.model is None:
            logger.info("Loading pre-trained ResNet model...")
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            self.model.eval()
            # Remove the last fully connected layer
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
            
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Extract image features."""
        if not self.image_column or self.image_column not in X.columns:
            return np.array([]).reshape(len(X), 0)
            
        image_paths = X[self.image_column].fillna('')
        
        if not self.use_pretrained:
            # Return dummy features if not using pre-trained model
            return np.random.rand(len(image_paths), 512)
            
        # Extract features using pre-trained model
        features = []
        with torch.no_grad():
            for img_path in image_paths:
                img = self._load_image(img_path)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()
                    self.model = self.model.cuda()
                feat = self.model(img_tensor).squeeze().cpu().numpy()
                features.append(feat)
                
        return np.vstack(features) if features else np.array([]).reshape(len(X), 0)

class FeatureEngineer:
    """Main feature engineering pipeline."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the feature engineer with configuration."""
        self.config = config or {}
        self.text_extractor = None
        self.image_extractor = None
        self.scaler = StandardScaler()
        self.target_encoders = {}
        self.feature_types = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature engineering pipeline."""
        self.feature_types = {
            'numeric': X.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical': X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(),
            'text': ['catalog_content'] if 'catalog_content' in X.columns else [],
            'image': ['image_link'] if 'image_link' in X.columns else []
        }
        
        # Remove text and image features from categorical features
        self.feature_types['categorical'] = [
            f for f in self.feature_types['categorical'] 
            if f not in self.feature_types['text'] + self.feature_types['image']
        ]
        
        # Initialize feature extractors
        if self.feature_types['text']:
            self.text_extractor = TextFeatureExtractor(
                self.feature_types['text'],
                use_bert=self.config.get('use_bert', False)
            )
            self.text_extractor.fit(X)
            
        if self.feature_types['image']:
            self.image_extractor = ImageFeatureExtractor(
                self.feature_types['image'][0],
                use_pretrained=self.config.get('use_pretrained_cnn', True)
            )
            self.image_extractor.fit(X)
            
        # Fit target encoders for categorical features
        if y is not None and self.feature_types['categorical']:
            for col in self.feature_types['categorical']:
                self.target_encoders[col] = TargetEncoder()
                self.target_encoders[col].fit(X[col], y)
                
        # Fit scaler for numeric features
        if self.feature_types['numeric']:
            self.scaler.fit(X[self.feature_types['numeric']])
            
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform the input data into features."""
        features = []
        
        # Process numeric features
        if self.feature_types['numeric']:
            num_features = self.scaler.transform(X[self.feature_types['numeric']])
            features.append(num_features)
            
        # Process categorical features
        if self.feature_types['categorical'] and self.target_encoders:
            cat_features = []
            for col in self.feature_types['categorical']:
                if col in self.target_encoders:
                    encoded = self.target_encoders[col].transform(X[col])
                    cat_features.append(encoded.values.reshape(-1, 1))
            if cat_features:
                features.append(np.hstack(cat_features))
                
        # Process text features
        if self.text_extractor is not None:
            text_features = self.text_extractor.transform(X)
            features.append(text_features)
            
        # Process image features
        if self.image_extractor is not None:
            image_features = self.image_extractor.transform(X)
            features.append(image_features)
            
        # Combine all features
        if not features:
            return np.array([]).reshape(len(X), 0)
            
        return np.hstack(features)
