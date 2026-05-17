# Price Prediction Challenge - Solution Documentation

## Team: 404 Not Found yet
**Submission Date:** October 13, 2025  
**SMAPE Score:** 45.75%

## Team Members
1. Rohit Kamatam
2. Shiva Krishna
3. Sarai Monisha
4. Rithika Kylasa

## Table of Contents
1. [Solution Overview](#solution-overview)
2. [Feature Engineering](#feature-engineering)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Results](#results)
6. [How to Run](#how-to-run)
7. [Dependencies](#dependencies)
8. [Future Improvements](#future-improvements)

## Solution Overview
This solution implements an enhanced ensemble approach for price prediction, achieving a SMAPE score of 45.75%. The pipeline includes advanced feature engineering, multiple model training, and ensemble prediction to improve accuracy and robustness.

## Feature Engineering

### Text Processing
- Extracted TF-IDF features with 800 dimensions (1-3 n-grams)
- Added CountVectorizer features with 400 dimensions (1-2 n-grams)
- Processed text to extract various linguistic features

### Numerical Features
- **IPQ (Items per Quantity):** Extracted from product descriptions using multiple patterns
- **Volume/Weight/Dimensions:** Extracted and normalized
- **Brand Statistics:** Mean, median, std, count, min, max prices by brand
- **Text Statistics:** Length, word count, sentence count, etc.
- **Interaction Features:** Combined features to capture relationships

### Categorical Encoding
- Target encoding for brand information
- Frequency encoding for categorical variables
- Handling of rare categories

## Model Architecture

### Ensemble Models
1. **XGBoost**
   - 1200 trees, max depth 8
   - Learning rate: 0.04
   - Subsampling and column sampling at 0.8
   - L1/L2 regularization (alpha=0.1, lambda=0.1)

2. **LightGBM**
   - 1200 trees, max depth 8
   - Learning rate: 0.04
   - Feature fraction: 0.8
   - Regularization parameters tuned

3. **Random Forest**
   - 300 trees, max depth 10
   - Minimum samples split: 5
   - Minimum samples leaf: 2

4. **Ridge Regression**
   - L2 regularization (alpha=1.0)
   - Features scaled using StandardScaler

### Ensemble Strategy
- Simple averaging of model predictions
- Log transformation of target variable for training
- Exponential transformation of predictions

## Training Process
- 5-fold cross-validation for model evaluation
- Early stopping based on validation performance
- Separate preprocessing for text and numerical features
- Feature scaling for linear models

## Results

### Cross-Validation Performance
- **SMAPE:** 45.75%
- **Model Contributions:**
  - XGBoost: Primary predictor
  - LightGBM: Secondary predictor with different split criteria
  - Random Forest: Captures non-linear relationships
  - Ridge Regression: Provides robust baseline



## How to Run

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Run the improved solution
python improved_ultra_fast_solution.py
```

### Expected Output
- Training progress and validation metrics
- Cross-validation results
- Final predictions saved to `test_out_improved.csv`

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- scipy

## Future Improvements
1. **Feature Engineering**
   - More sophisticated text embeddings (BERT, Sentence-Transformers)
   - Image feature extraction from product images
   - Advanced feature interactions

2. **Modeling**
   - Neural network architectures
   - Stacking with meta-learner
   - Hyperparameter optimization

3. **Ensemble**
   - Weighted averaging based on model confidence
   - Dynamic model selection
   - Blending with different validation strategies

4. **Deployment**
   - API for real-time predictions
   - Monitoring and retraining pipeline
   - A/B testing framework

---
*Last updated: October 13, 2025*
