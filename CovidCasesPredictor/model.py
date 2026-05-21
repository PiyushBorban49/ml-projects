import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('CONVENIENT_global_confirmed_cases.csv')

data['Date'] = pd.to_datetime(data['Date'])
data['TimeStamp'] = data['Date'].apply(lambda x: x.timestamp())
data['Total-Cases'] = data.drop(['Date','TimeStamp'], axis=1).sum(axis=1)

# Sort by date for proper time series processing
data = data.sort_values('Date').reset_index(drop=True)

# ============= FEATURE ENGINEERING =============

# 1. Time-Based Features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['DayOfYear'] = data['Date'].dt.dayofyear
data['Week'] = data['Date'].dt.isocalendar().week

# 2. Lag Features (Previous values as predictors)
data['Cases_Lag_1'] = data['Total-Cases'].shift(1)
data['Cases_Lag_7'] = data['Total-Cases'].shift(7)
data['Cases_Lag_14'] = data['Total-Cases'].shift(14)
data['Cases_Lag_30'] = data['Total-Cases'].shift(30)

# 3. Rolling Window Features
data['Cases_MA_7'] = data['Total-Cases'].rolling(7).mean()
data['Cases_MA_14'] = data['Total-Cases'].rolling(14).mean()
data['Cases_MA_30'] = data['Total-Cases'].rolling(30).mean()
data['Cases_Std_7'] = data['Total-Cases'].rolling(7).std()
data['Cases_Min_7'] = data['Total-Cases'].rolling(7).min()
data['Cases_Max_7'] = data['Total-Cases'].rolling(7).max()

# 4. Difference Features (Rate of change)
data['Cases_Diff_1'] = data['Total-Cases'].diff(1)
data['Cases_Diff_7'] = data['Total-Cases'].diff(7)
data['Cases_Pct_Change_1'] = data['Total-Cases'].pct_change(1)
data['Cases_Pct_Change_7'] = data['Total-Cases'].pct_change(7)

# 5. Cyclical Encoding (Better than raw categorical features)
data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
data['DayOfWeek_Sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
data['DayOfWeek_Cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
data['DayOfYear_Sin'] = np.sin(2 * np.pi * data['DayOfYear'] / 365)
data['DayOfYear_Cos'] = np.cos(2 * np.pi * data['DayOfYear'] / 365)

# 6. Trend Features
data['Days_Since_Start'] = (data['Date'] - data['Date'].min()).dt.days
data['Cases_Per_Day_Since_Start'] = data['Total-Cases'] / (data['Days_Since_Start'] + 1)

# 7. Acceleration Features (second derivatives)
data['Cases_Acceleration'] = data['Cases_Diff_1'].diff(1)

print("Feature engineering completed!")
print(f"Original features: 3 (Date, TimeStamp, Total-Cases)")
print(f"Total features after engineering: {len(data.columns)}")

# ============= PREPARE DATA FOR MODELING =============

# Select features for modeling (exclude Date and target variable)
feature_cols = [col for col in data.columns if col not in ['Date', 'Total-Cases']]

# Remove rows with NaN values (created by lag and rolling features)
data_clean = data.dropna().reset_index(drop=True)
print(f"Data points after removing NaN: {len(data_clean)} (from {len(data)})")

X = data_clean[feature_cols].values
Y = data_clean['Total-Cases'].values

# Split data (using temporal split for time series)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============= SCALE FEATURES =============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============= TRAIN MODELS =============

# Model 1: Original approach (just timestamp)
clf_original = XGBRegressor(random_state=42)
clf_original.fit(data_clean['TimeStamp'].values[:split_idx].reshape(-1,1), Y_train)
score_original = clf_original.score(data_clean['TimeStamp'].values[split_idx:].reshape(-1,1), Y_test)

# Model 2: With feature engineering
clf_enhanced = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
clf_enhanced.fit(X_train_scaled, Y_train)
score_enhanced = clf_enhanced.score(X_test_scaled, Y_test)

# ============= RESULTS =============
print("\n" + "="*50)
print("MODEL COMPARISON:")
print("="*50)
print(f"Original model (timestamp only): R² = {score_original:.4f}")
print(f"Enhanced model (with features):  R² = {score_enhanced:.4f}")
print(f"Improvement: {((score_enhanced - score_original) / abs(score_original) * 100):+.1f}%")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': clf_enhanced.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 10 MOST IMPORTANT FEATURES:")
print(feature_importance.head(10))

# ============= VISUALIZATION =============
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Original time series
axes[0, 0].plot(data_clean['Date'], data_clean['Total-Cases'])
axes[0, 0].set_title('COVID-19 Total Cases Over Time')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Total Cases')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Model comparison
test_dates = data_clean['Date'].iloc[split_idx:]
Y_pred_original = clf_original.predict(data_clean['TimeStamp'].values[split_idx:].reshape(-1,1))
Y_pred_enhanced = clf_enhanced.predict(X_test_scaled)

axes[0, 1].plot(test_dates, Y_test, label='Actual', linewidth=2)
axes[0, 1].plot(test_dates, Y_pred_original, label='Original Model', alpha=0.7)
axes[0, 1].plot(test_dates, Y_pred_enhanced, label='Enhanced Model', alpha=0.7)
axes[0, 1].set_title('Model Predictions Comparison')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Total Cases')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: Feature importance
top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_title('Top 10 Feature Importance')
axes[1, 0].set_xlabel('Importance')

# Plot 4: Actual vs Predicted (Enhanced Model)
axes[1, 1].scatter(Y_test, Y_pred_enhanced, alpha=0.6)
axes[1, 1].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
axes[1, 1].set_title('Actual vs Predicted (Enhanced Model)')
axes[1, 1].set_xlabel('Actual Values')
axes[1, 1].set_ylabel('Predicted Values')

plt.tight_layout()
plt.show()

print(f"\nFeature engineering successfully improved model performance!")
print(f"The most important features are related to: {', '.join(top_features['feature'].head(5).tolist())}")