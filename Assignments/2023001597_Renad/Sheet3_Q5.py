# Q5 (Sheet 3 )– Effect of Decision Threshold

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predict probabilities
prob = model.predict_proba(X_test)[:, 1]   # probability of class "1 - benign"

# Try different thresholds
thresholds = [0.5, 0.6, 0.7, 0.8]

for t in thresholds:
    y_pred_t = (prob >= t).astype(int)
    acc = accuracy_score(y_test, y_pred_t)
    cm = confusion_matrix(y_test, y_pred_t)

    print(f"\n=== Threshold: {t} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
