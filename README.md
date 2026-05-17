# Price Prediction Challenge

## Problem Statement

The challenge involves predicting product prices based on various product features including:
- Product descriptions and text data
- Brand information
- Numerical attributes (volume, weight, dimensions)
- Items per quantity (IPQ) information

The goal is to build a machine learning model that can accurately predict prices for products using the provided features, with performance evaluated using the SMAPE (Symmetric Mean Absolute Percentage Error) metric.

## Our Approach

Our solution implements an enhanced ensemble approach for price prediction, achieving a SMAPE score of **45.75%**. The methodology includes:

### Feature Engineering
- **Text Processing**: TF-IDF features (800 dimensions with 1-3 n-grams) and CountVectorizer features (400 dimensions with 1-2 n-grams)
- **Numerical Features**: 
  - IPQ (Items per Quantity) extraction from product descriptions
  - Volume, weight, and dimension extraction and normalization
  - Brand statistics (mean, median, std, count, min, max prices by brand)
  - Text statistics (length, word count, sentence count)
  - Interaction features to capture relationships
- **Categorical Encoding**: Target encoding for brand information and frequency encoding for categorical variables

### Model Architecture
We use an ensemble of four models:
1. **XGBoost**: 1200 trees, max depth 8, learning rate 0.04 with L1/L2 regularization
2. **LightGBM**: 1200 trees, max depth 8, learning rate 0.04 with feature fraction 0.8
3. **Random Forest**: 300 trees, max depth 10 for capturing non-linear relationships
4. **Ridge Regression**: L2 regularization with StandardScaler for robust baseline

### Ensemble Strategy
- Simple averaging of model predictions
- Log transformation of target variable for training
- Exponential transformation of predictions
- 5-fold cross-validation for model evaluation

## What is SMAPE?

**SMAPE (Symmetric Mean Absolute Percentage Error)** is an evaluation metric used to measure the accuracy of predictions. It is calculated as:

```
SMAPE = (100% / n) * Σ(|F_t - A_t| / ((|A_t| + |F_t|) / 2))
```

Where:
- `F_t` = Forecast value (predicted price)
- `A_t` = Actual value (true price)
- `n` = Number of observations

### Key Characteristics:
- **Symmetric**: Treats overestimation and underestimation equally
- **Scale-independent**: Works well with different scales of data
- **Bounded**: Ranges from 0% to 200%, with lower values indicating better predictions
- **Percentage-based**: Easy to interpret as a percentage error

### Why SMAPE?
Unlike traditional MAPE (Mean Absolute Percentage Error), SMAPE handles cases where actual values are close to zero better, as it uses both actual and predicted values in the denominator. This makes it more suitable for price prediction tasks where prices can vary significantly in magnitude.

## Team
- **Team Name**: 404 Not Found yet
- **Members**: Rohit Kamatam, Shiva Krishna, Sarai Monisha, Rithika Kylasa
- **Submission Date**: October 13, 2025

## Installation and Usage

To set up the environment and run the solution, follow these steps:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the solution script:
   ```bash
   python improved_ultra_fast_solution.py
   ```

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the solution
python improved_ultra_fast_solution.py
```

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- scipy
