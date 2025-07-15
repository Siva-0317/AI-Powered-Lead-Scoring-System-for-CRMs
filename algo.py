import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import joblib

# Load the dataset
file_path = 'data/cleaned_lead_scoring.csv'
data = pd.read_csv(file_path)
#data=data.drop(columns=['Prospect ID','Lead Number'])
# Drop unnecessary columns
#data = data.drop(columns=['Patient Id'])

# Separate features and target variable
target = 'Converted'
X = data.drop(columns=[target])
y = data[target]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

# OneHotEncode the target variable
target_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
y_encoded = target_encoder.fit_transform(y.values.reshape(-1, 1))

# Convert OneHotEncoded target back to numerical labels
y_encoded_labels = np.argmax(y_encoded, axis=1)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Algorithms to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', use_label_encoder=False)
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded_labels, test_size=0.2, random_state=42)

# Evaluate models
results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}

# Display results
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values(by='R2', ascending=False)
print("Model Performance:")
print(results_df)

# Best model selection
best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\nBest Model: {best_model_name}")

# Train final model with best algorithm
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
final_pipeline.fit(X_train, y_train)