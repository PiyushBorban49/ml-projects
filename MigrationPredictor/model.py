import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import svm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor

# Load data
data = pd.read_csv('migration_nz.csv')

# Encode categorical variables
data['Measure'].replace("Arrivals", 0, inplace=True)
data['Measure'].replace("Departures", 1, inplace=True)
data['Measure'].replace("Net", 2, inplace=True)

data['CountryID'] = pd.factorize(data.Country)[0]
data['CitID'] = pd.factorize(data.Citizenship)[0]

# Handle missing values
data["Value"].fillna(data["Value"].median(), inplace=True)

# Drop original categorical columns
data.drop('Country', axis=1, inplace=True)
data.drop('Citizenship', axis=1, inplace=True)

# Preparing data
X = data[['CountryID', 'Measure', 'Year', 'CitID']].values
Y = data['Value'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=9)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=70, max_features=3, max_depth=5, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
rf_score = rf.score(X_test, y_test)
rf_predictions = rf.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)

print("Random Forest Results:")
print(f"R² Score: {rf_score:.4f}")
print(f"Mean Squared Error: {rf_mse:.4f}")
print(f"RMSE: {np.sqrt(rf_mse):.4f}")

# Feature importance
feature_names = ['CountryID', 'Measure', 'Year', 'CitID']
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)

# Optional: Compare with other models
print("\n" + "="*50)
print("Comparing with other models:")

# XGBoost
xgb = XGBRegressor(n_estimators=70, max_depth=5, random_state=42)
xgb.fit(X_train, y_train)
xgb_score = xgb.score(X_test, y_test)
xgb_predictions = xgb.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_predictions)

print(f"XGBoost R² Score: {xgb_score:.4f}")
print(f"XGBoost MSE: {xgb_mse:.4f}")

# Create visualization
plt.figure(figsize=(12, 8))

# Feature importance plot
plt.subplot(2, 2, 1)
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.title('Random Forest Feature Importance')
plt.xlabel('Importance')

# Actual vs Predicted for Random Forest
plt.subplot(2, 2, 2)
plt.scatter(y_test, rf_predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest: Actual vs Predicted')

# Residuals plot
plt.subplot(2, 2, 3)
residuals = y_test - rf_predictions
plt.scatter(rf_predictions, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Random Forest Residuals Plot')

# Model comparison
plt.subplot(2, 2, 4)
models = ['Random Forest', 'XGBoost']
scores = [rf_score, xgb_score]
plt.bar(models, scores)
plt.ylabel('R² Score')
plt.title('Model Comparison')
plt.ylim(0, 1)

plt.tight_layout()
plt.show()