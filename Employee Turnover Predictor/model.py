import optuna
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import optuna.visualization as vis
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data = pd.read_csv('datasets_9768_13874_HR_comma_sep.csv')

# converting columns satisfaction_level and last_evaluation in percentage
data['satisfaction_level'] = data['satisfaction_level'].apply(lambda x: x*100)
data['last_evaluation'] = data['last_evaluation'].apply(lambda x: x*100)

# ordinal encoding on column salary
ord = OrdinalEncoder(categories=[['low','medium','high']])
data['salary'] = ord.fit_transform(data[['salary']])

# taking sales as nominal
encoded = pd.get_dummies(data[['sales']],drop_first=True).astype(int)
data = data.drop(['sales'],axis=1)
data = pd.concat([data,encoded],axis=1)

# preparing data
X = data.drop(['left'],axis=1).values
Y = data['left'].values

# using randomoversample do that classes get equal opportunities
ros = RandomOverSampler()
X_resampled,Y_resampled = ros.fit_resample(X,Y)

# spliting data
X_train,X_test,Y_train,Y_test = train_test_split(X_resampled,Y_resampled,test_size=0.2)

# using standard scaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# using decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))



# using hyperparameter tuning
def objective(trial):
    max_depth = trial.suggest_int('max_depth',1,50)
    min_samples_split = trial.suggest_int('min_samples_split',2,20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf',1,10)

    model = DecisionTreeClassifier(max_depth=max_depth,min_samples_split=min_samples_split)

    score = cross_val_score(model,X_train,Y_train,cv=10,scoring='accuracy')

    return score.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=100)

print(study.best_params)

















# 1. Optimization History Plot
fig_history = vis.plot_optimization_history(study)
fig_history.update_layout(title="Optimization History - Accuracy over Trials")
fig_history.show()

# 2. Parameter Importance Plot
fig_importance = vis.plot_param_importances(study)
fig_importance.update_layout(title="Parameter Importance")
fig_importance.show()

# 3. Parallel Coordinate Plot
fig_parallel = vis.plot_parallel_coordinate(study)
fig_parallel.update_layout(title="Parallel Coordinate Plot")
fig_parallel.show()

# 4. Contour Plot (for parameter relationships)
if len(study.best_params) >= 2:
    param_names = list(study.best_params.keys())
    for i in range(len(param_names)):
        for j in range(i+1, len(param_names)):
            fig_contour = vis.plot_contour(study, params=[param_names[i], param_names[j]])
            fig_contour.update_layout(title=f"Contour Plot: {param_names[i]} vs {param_names[j]}")
            fig_contour.show()

# 5. Slice Plot
fig_slice = vis.plot_slice(study)
fig_slice.update_layout(title="Slice Plot - Parameter Effects")
fig_slice.show()

# CUSTOM MATPLOTLIB VISUALIZATIONS
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Optuna Hyperparameter Optimization Results', fontsize=16, fontweight='bold')

# 1. Trial values over time
trial_numbers = [trial.number for trial in study.trials]
trial_values = [trial.value for trial in study.trials if trial.value is not None]
trial_nums_with_values = [trial.number for trial in study.trials if trial.value is not None]

axes[0, 0].plot(trial_nums_with_values, trial_values, 'b-', alpha=0.7, linewidth=1)
axes[0, 0].scatter(trial_nums_with_values, trial_values, c='blue', alpha=0.6, s=20)
axes[0, 0].axhline(y=study.best_value, color='red', linestyle='--',
                   label=f'Best Score: {study.best_value:.4f}')
axes[0, 0].set_xlabel('Trial Number')
axes[0, 0].set_ylabel('Cross-Validation Accuracy')
axes[0, 0].set_title('Optimization Progress')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Parameter distribution - max_depth
max_depths = [trial.params.get('max_depth') for trial in study.trials if trial.value is not None]
axes[0, 1].hist(max_depths, bins=20, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].axvline(x=study.best_params['max_depth'], color='red', linestyle='--',
                   label=f'Best: {study.best_params["max_depth"]}')
axes[0, 1].set_xlabel('Max Depth')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Max Depth Values')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Parameter distribution - min_samples_split
min_samples_splits = [trial.params.get('min_samples_split') for trial in study.trials if trial.value is not None]
axes[1, 0].hist(min_samples_splits, bins=15, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].axvline(x=study.best_params['min_samples_split'], color='red', linestyle='--',
                   label=f'Best: {study.best_params["min_samples_split"]}')
axes[1, 0].set_xlabel('Min Samples Split')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Min Samples Split Values')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Parameter distribution - min_samples_leaf
min_samples_leafs = [trial.params.get('min_samples_leaf') for trial in study.trials if trial.value is not None]
axes[1, 1].hist(min_samples_leafs, bins=10, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axvline(x=study.best_params['min_samples_leaf'], color='red', linestyle='--',
                   label=f'Best: {study.best_params["min_samples_leaf"]}')
axes[1, 1].set_xlabel('Min Samples Leaf')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Min Samples Leaf Values')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional scatter plot showing relationships
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# Scatter plot: max_depth vs accuracy
ax[0].scatter(max_depths, trial_values, alpha=0.6, c='blue', s=30)
ax[0].set_xlabel('Max Depth')
ax[0].set_ylabel('Cross-Validation Accuracy')
ax[0].set_title('Max Depth vs Accuracy')
ax[0].grid(True, alpha=0.3)

# Scatter plot: min_samples_split vs accuracy
ax[1].scatter(min_samples_splits, trial_values, alpha=0.6, c='green', s=30)
ax[1].set_xlabel('Min Samples Split')
ax[1].set_ylabel('Cross-Validation Accuracy')
ax[1].set_title('Min Samples Split vs Accuracy')
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary statistics
print("\n" + "="*50)
print("OPTIMIZATION SUMMARY")
print("="*50)
print(f"Total trials: {len(study.trials)}")
print(f"Completed trials: {len([t for t in study.trials if t.value is not None])}")
print(f"Failed trials: {len(study.trials) - len([t for t in study.trials if t.value is not None])}")
print(f"Best trial number: {study.best_trial.number}")
print(f"Best parameters: {study.best_params}")
print(f"Best cross-validation score: {study.best_value:.4f}")


