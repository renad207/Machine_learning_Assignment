# Q4  (Sheet 4) – Models Comparison 

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


# Load Dataset

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 1) Decision Tree (Full)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt.predict(X_test))


# 2) Random Forest

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

# Top 5 features
rf_importances = (
    pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    })
    .sort_values("Importance", ascending=False)
    .head(5)
)


# 3) Gradient Boosting

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

gb_train_acc = accuracy_score(y_train, gb.predict(X_train))
gb_test_acc = accuracy_score(y_test, gb.predict(X_test))

# Top 5 features
gb_importances = (
    pd.DataFrame({
        "Feature": X.columns,
        "Importance": gb.feature_importances_
    })
    .sort_values("Importance", ascending=False)
    .head(5)
)


# Print Results

print("\n=== Training & Testing Accuracy ===")
print(f"Decision Tree (Full)     -> Train: {dt_train_acc:.4f} | Test: {dt_test_acc:.4f}")
print(f"Random Forest            -> Train: {rf_train_acc:.4f} | Test: {rf_test_acc:.4f}")
print(f"Gradient Boosting        -> Train: {gb_train_acc:.4f} | Test: {gb_test_acc:.4f}")

print("\n=== Top 5 Features – Random Forest ===")
print(rf_importances)

print("\n=== Top 5 Features – Gradient Boosting ===")
print(gb_importances)
