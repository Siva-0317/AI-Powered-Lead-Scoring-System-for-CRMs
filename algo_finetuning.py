import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

# Load cleaned dataset
df = pd.read_csv("data/cleaned_lead_scoring.csv")

# Encode categorical features
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & target
X = df.drop("Converted", axis=1)
y = df["Converted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define parameter grid
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# XGBoost model
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

# GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=3,
    verbose=2,
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Print best model and classification report
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
print("Best ROC AUC Score (CV):", grid_search.best_score_)

y_pred = best_model.predict(X_test)
print("\nClassification Report on Test Set:\n")
print(classification_report(y_test, y_pred))

# Visualization of feature importances
import matplotlib.pyplot as plt
from xgboost import plot_importance

plot_importance(best_model, max_num_features=10)
plt.tight_layout()
plt.show()

#save the model
import joblib
joblib.dump(best_model, "xgboost_lead_model.pkl")

#save label encoders
# Save label encoders
import pickle

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

import pickle

with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

print("Available columns in label_encoders:", list(encoders.keys()))




