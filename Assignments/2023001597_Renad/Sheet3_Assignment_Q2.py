# Task (Sheet 3) Q2 – Feature Importance for Logistic Regression

import pandas as pd
import numpy as np

# Extract model coefficients
coeffs = pipeline.named_steps['model'].coef_[0]

# Put into dataframe
feature_importance = pd.DataFrame({
    'Feature': data.feature_names,
    'Importance': np.abs(coeffs)
}).sort_values('Importance', ascending=False)

# Show top 5 features
print(feature_importance.head())