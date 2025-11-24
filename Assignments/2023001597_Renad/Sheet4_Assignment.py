# Assignment (Sheet 4 )

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 1) Decision Tree (FULL)

dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

dt_full_train = accuracy_score(y_train, dt_full.predict(X_train))
dt_full_test  = accuracy_score(y_test, dt_full.predict(X_test))


# 2) Decision Tree (Pruned)

dt_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_pruned.fit(X_train, y_train)

dt_pruned_train = accuracy_score(y_train, dt_pruned.predict(X_train))
dt_pruned_test  = accuracy_score(y_test, dt_pruned.predict(X_test))


# 3) Random Forest

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_train = accuracy_score(y_train, rf.predict(X_train))
rf_test  = accuracy_score(y_test, rf.predict(X_test))


# 4) Gradient Boosting

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

gb_train = accuracy_score(y_train, gb.predict(X_train))
gb_test  = accuracy_score(y_test, gb.predict(X_test))


# Print results

print("\n=== Model Comparison ===")
print(f"Decision Tree (Full)     -> Train: {dt_full_train:.4f} | Test: {dt_full_test:.4f}")
print(f"Decision Tree (Pruned)   -> Train: {dt_pruned_train:.4f} | Test: {dt_pruned_test:.4f}")
print(f"Random Forest            -> Train: {rf_train:.4f} | Test: {rf_test:.4f}")
print(f"Gradient Boosting        -> Train: {gb_train:.4f} | Test: {gb_test:.4f}")
