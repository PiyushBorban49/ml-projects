import numpy as np
import optuna
import optuna_integration
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

data = pd.read_csv('online_shoppers_intention.csv')

data['Month'] = pd.factorize(data['Month'])[0]
onehot = pd.get_dummies(data['VisitorType'],drop_first=True).astype(int)
data = pd.concat([data,onehot],axis=1)

data['Weekend'] = data['Weekend'].astype(int)
data['Revenue'] = data['Revenue'].astype(int)

data = data.drop(['VisitorType'],axis=1)

X = data.drop('Revenue',axis=1).values
Y = data['Revenue'].values

ros = RandomOverSampler()
X_resampled , Y_resampled = ros.fit_resample(X,Y)

X_train,X_test,Y_train,Y_test = train_test_split(X_resampled,Y_resampled,test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# hyperparameter tuning
def ojective(trial):
    n_estimators = trial.suggest_int('n_estimators',50,200)
    max_depth = trial.suggest_int('max_depth',10,100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,min_samples_split=min_samples_split)
    score = cross_val_score(model,X_train_scaled,Y_train,cv=10,scoring="accuracy")
    return score.mean()

study = optuna.create_study(direction="maximize")
study.optimize(ojective,n_trials=100)

print(study.best_params)


clf = RandomForestClassifier()
clf.fit(X_train_scaled,Y_train)
print(clf.score(X_test_scaled,Y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_resampled, Y_resampled, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {scores}")
print(f"Mean CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")


feature_names = data.drop('Revenue', axis=1).columns.tolist()
feature_importances = clf.feature_importances_

print(f"Number of features: {len(feature_names)}")
print(f"Number of importances: {len(feature_importances)}")

important_features = pd.DataFrame({
    'features': feature_names,
    'importance': feature_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(important_features.head(10))

plt.figure(figsize=(10, 8))
top_10_features = important_features.head(10)
plt.barh(range(len(top_10_features)), top_10_features['importance'].values)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Top 10 Feature Importance from Random Forest')
plt.yticks(range(len(top_10_features)), top_10_features['features'].values)

for i, importance in enumerate(top_10_features['importance'].values):
    plt.text(importance, i, f'{importance:.3f}',va='center', ha='left', fontsize=10)

plt.tight_layout()
plt.show()