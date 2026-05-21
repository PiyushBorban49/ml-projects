import random
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from pandas.core.common import random_state
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from torch.utils.data import RandomSampler

data = pd.read_csv("final_dataset.csv")
data = data.dropna(subset=["beds","baths","size","price"])

X = data[["beds","baths","size"]].values.astype(int)
Y = data["price"].values.astype(int)

ros = RandomOverSampler(random_state=42)
X_resampled,Y_resampled = ros.fit_resample(X,Y)

X_train,X_test,Y_train,Y_test = train_test_split(X_resampled,Y_resampled,test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def variance(y):
    """Calculate variance for regression splitting criterion"""
    if len(y) == 0:
        return 0
    return np.var(y)


class Node:
    def __init__(self, features=None, threshold=None, left=None, right=None, *, value=None):
        self.left = left
        self.right = right
        self.features = features
        self.threshold = threshold
        self.value = value

    def isLeafNode(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features

    def fit(self, X, Y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, Y)

    def _grow_tree(self, X, Y, depth=0):
        n_samples, n_feature = X.shape

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(Y)) == 1:
            leaf_value = np.mean(Y)
            return Node(value=leaf_value)

        feats_index = np.random.choice(n_feature, self.n_features, replace=False)

        best_features, best_threshold = self._best_criteria(X, Y, feats_index)

        # If no good split found
        if best_features is None:
            leaf_value = np.mean(Y)
            return Node(value=leaf_value)

        left_index, right_index = self._split(X[:, best_features], best_threshold)

        # If split doesn't actually divide the data
        if len(left_index) == 0 or len(right_index) == 0:
            leaf_value = np.mean(Y)
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_index, :], Y[left_index], depth + 1)
        right = self._grow_tree(X[right_index, :], Y[right_index], depth + 1)
        return Node(best_features, best_threshold, left, right)

    def _best_criteria(self, X, Y, feats_index):
        best_gain = -1
        split_features, split_threshold = None, None

        for feats in feats_index:
            X_col = X[:, feats]
            thresholds = np.unique(X_col)

            # Limit number of thresholds to check for efficiency
            if len(thresholds) > 10:
                thresholds = np.percentile(X_col, np.linspace(10, 90, 10))

            for thr in thresholds:
                gain = self._variance_reduction(Y, X_col, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_threshold = thr
                    split_features = feats
        return split_features, split_threshold

    def _variance_reduction(self, Y, X_col, thr):
        """Calculate variance reduction for regression"""
        parent_variance = variance(Y)

        left_index, right_index = self._split(X_col, thr)

        if len(left_index) == 0 or len(right_index) == 0:
            return 0

        n = len(Y)
        n_l, n_r = len(left_index), len(right_index)
        v_l, v_r = variance(Y[left_index]), variance(Y[right_index])
        child_variance = (n_l / n) * v_l + (n_r / n) * v_r
        return parent_variance - child_variance

    def _split(self, X_col, thr):
        return np.argwhere(X_col <= thr).flatten(), np.argwhere(X_col > thr).flatten()

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.isLeafNode():
            return node.value

        if x[node.features] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


clf = DecisionTree(max_depth=10)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

# Proper evaluation metrics for regression
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

print("Regression Evaluation Metrics:")
print(f"Mean Absolute Error: {mean_absolute_error(Y_test, predictions):.2f}")
print(f"Root Mean Squared Error: {root_mean_squared_error(Y_test, predictions):.2f}")
print(f"R-squared Score: {r_squared(Y_test, predictions):.4f}")

# Optional: Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, predictions, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
