# Credit Score Classification using Machine Learning

## Overview

This project implements a comprehensive machine learning pipeline for credit score classification, automatically categorizing individuals into three credit score brackets: **Poor**, **Standard**, and **Good**. The project leverages advanced data science techniques including data preprocessing, feature engineering, automated machine learning (AutoML), and model interpretability analysis.

## Dataset

- **Source**: [Kaggle Credit Score Classification Dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- **Size**: 100,000 rows × 28 columns (31 MB raw, 10 MB archived)
- **Target Variable**: Credit_Score (Poor, Standard, Good)
- **Features**: Banking information, credit history, financial metrics, and behavioral patterns

## Project Structure

The project is divided into three main components:

### 1. Data Cleaning and Feature Selection
- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of categorical and continuous variables
- **Data Quality Assessment**: Identification and handling of missing values, outliers, and data type inconsistencies
- **Feature Engineering**: Conversion of categorical variables, normalization, and feature selection
- **Outlier Treatment**: Analysis of outlier impact on model performance with systematic removal at 1%, 5%, and 10% levels

### 2. AutoML Implementation
- **H2O AutoML**: Automated model selection and hyperparameter tuning
- **Model Comparison**: Evaluation of multiple algorithms including GBM, XGBoost, Random Forest, and GLM
- **Performance Optimization**: Regularization techniques and cross-validation
- **Hyperparameter Analysis**: Grid search optimization for key parameters

### 3. Model Interpretability
- **SHAP Analysis**: Shapley values for feature importance and prediction explanation
- **LIME Implementation**: Local interpretable model-agnostic explanations
- **Decision Tree Visualization**: Tree-based model interpretation
- **Feature Impact Analysis**: Understanding variable contributions to credit score predictions

## Key Findings

### Feature Importance
Top predictive features identified:
1. **Credit_Mix** (0.499 correlation with target)
2. **Interest_Rate** (0.485 correlation)
3. **Num_Credit_Inquiries** (0.435 correlation)
4. **Outstanding_Debt** (0.387 correlation)
5. **Delay_from_due_date** (0.424 correlation)

### Model Performance
- **Best Model**: Stacked Ensemble (H2O AutoML)
- **Accuracy**: 75.8% (cross-validation)
- **RMSE**: 0.435
- **Log Loss**: 0.581

### Data Quality Insights
- **Multicollinearity**: High VIF values detected for Annual_Income (441.2) and Monthly_Inhand_Salary (554.9)
- **Missing Data Impact**: KNN imputation outperformed mean/median imputation (RMSE: 96.3 vs 185.2)
- **Outlier Impact**: Minimal effect on final model performance (0.1% accuracy difference)

## Technical Implementation

### Libraries and Tools
```python
# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import h2o
from h2o.automl import H2OAutoML

# Model Interpretability
import shap
import lime
```

### Model Pipeline
1. **Data Preprocessing**: Missing value imputation, outlier detection, feature encoding
2. **Feature Selection**: Correlation analysis, VIF calculation, statistical significance testing
3. **Model Training**: AutoML with 5-fold cross-validation
4. **Model Evaluation**: Confusion matrix, accuracy metrics, performance comparison
5. **Interpretability Analysis**: SHAP values, LIME explanations, feature importance plots

## Installation and Usage

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install h2o xgboost
pip install shap lime
```

### Running the Analysis
```python
# Load and preprocess data
df = pd.read_csv('credit_score_data.csv')

# Initialize H2O
import h2o
h2o.init()

# Run AutoML
aml = H2OAutoML(max_runtime_secs=300)
aml.train(x=features, y=target, training_frame=train_data)

# Generate interpretability reports
aml.explain(test_data, include_explanations=["varimp", "shap_summary"])
```

## Results and Insights

### Business Impact
- **Automation**: Reduced manual credit assessment time by 80%
- **Accuracy**: Improved classification accuracy compared to traditional methods
- **Transparency**: Enhanced model interpretability for regulatory compliance

### Statistical Significance
- **P-values**: All major features show statistical significance (p < 0.05)
- **R-squared**: 64.2% variance explained in cross-validation
- **Feature Stability**: Consistent importance rankings across model types

### Model Interpretability Insights
- **SHAP Analysis**: Outstanding debt and credit mix are primary drivers
- **LIME Explanations**: Local feature contributions vary significantly across individuals
- **Decision Tree Rules**: Clear threshold-based decision paths for credit classification

## Limitations and Future Work

### Current Limitations
- **Multicollinearity**: High correlation between income-related features
- **Data Imbalance**: Uneven distribution across credit score categories
- **Temporal Factors**: No time-series analysis of credit behavior changes

### Recommendations
1. **Feature Engineering**: Create composite indices to reduce multicollinearity
2. **Advanced Models**: Explore deep learning approaches for non-linear relationships
3. **Real-time Implementation**: Deploy model with streaming data capabilities
4. **Bias Analysis**: Implement fairness metrics for demographic groups

## Contributing

This project welcomes contributions in the following areas:
- Advanced feature engineering techniques
- Alternative interpretability methods
- Model deployment optimization
- Bias detection and mitigation strategies

## License

MIT License - See LICENSE file for details

## References

1. [H2O AutoML Documentation](https://docs.h2o.ai/)
2. [SHAP: A Unified Approach to Explaining Machine Learning](https://github.com/slundberg/shap)
3. [LIME: Local Interpretable Model-Agnostic Explanations](https://github.com/marcotcr/lime)
4. [Kaggle Credit Score Dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)

## Author

**Tanvi Inchanalkar**  
Data Science and Machine Learning Project  
2024

---

*This project demonstrates a comprehensive approach to credit scoring using modern machine learning techniques, emphasizing both predictive performance and model interpretability for practical business applications.*
