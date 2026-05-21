import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("vehicles.csv")
data = data.dropna(subset=["DateTime","Vehicles"])

#get date
def get_dom(dt):
    return dt.day

#get week day
def get_weekday(dt):
    return dt.weekday()

#get hour
def get_hour(dt):
    return dt.hour

#get year
def get_year(dt):
    return dt.year

#get month
def get_month(dt):
    return dt.month

#get year day
def get_dayofyear(dt):
    return dt.dayofyear

#get year month
def get_weekofyear(dt):
    return dt.isocalendar().week

data['DateTime'] = data['DateTime'].map(pd.to_datetime)
data['date'] = data['DateTime'].map(get_dom)
data['weekday'] = data['DateTime'].map(get_weekday)
data['hour'] = data['DateTime'].map(get_hour)
data['month'] = data['DateTime'].map(get_month)
data['year'] = data['DateTime'].map(get_year)
data['dayofyear'] = data['DateTime'].map(get_dayofyear)
data['weekofyear'] = data['DateTime'].map(get_weekofyear)

data = data.drop(['DateTime'],axis=1)
X = data.drop(['Vehicles'],axis=1)
Y = data['Vehicles']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

X_train = X_train.values if hasattr(X_train, 'values') else X_train
X_test = X_test.values if hasattr(X_test, 'values') else X_test
Y_train = Y_train.values if hasattr(Y_train, 'values') else Y_train
Y_test = Y_test.values if hasattr(Y_test, 'values') else Y_test

class Node:
    def __init__(self,features=None,thresholds=None,left=None,right=None,*,value=None):
        self.value = value
        self.left = left
        self.right = right
        self.threshold = thresholds
        self.features = features

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=10,n_features=None,max_bins=20):
        self.max_bins = max_bins
        self.root = None
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.max_depth = max_depth

    def fit(self,X,y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth=0):
        n_samples,n_features=X.shape
        n_label = len(np.unique(y))
        if depth>=self.max_depth or n_label == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        feats_index = np.random.choice(n_features,self.n_features,replace=False)
        best_features,best_threshold = self._best_criteria(X,y,feats_index)
        if best_features is None:
            return Node(value=self._most_common_label(y))
        left_index,right_index = self._split(X[:,best_features],best_threshold)
        if len(left_index) == 0 or len(right_index)==0:
            return Node(value=self._most_common_label(y))
        left = self._grow_tree(X[left_index,:],y[left_index],depth+1)
        right = self._grow_tree(X[right_index,:],y[right_index],depth+1)
        return Node(best_features,best_threshold,left,right)

    def _most_common_label(self,y):
        if len(y) == 0:
            return 0
        return np.mean(y)

    def _best_criteria(self,X,y,feats_index):
        best_gain = -1
        split_features,split_thresholds=None,None
        for feats in feats_index:
            X_col = X[:,feats]
            if len(np.unique(X_col))>self.max_bins:
                threshold = np.percentile(X_col, np.linspace(0, 100, self.max_bins))
            else:
                threshold = np.unique(X_col)
            for thr in threshold:
                gain = self._information_gain(X_col,y,thr)
                if gain>best_gain:
                    best_gain = gain
                    split_thresholds = thr
                    split_features = feats
        return split_features,split_thresholds

    def _split(self,X_col,thr):
        return np.argwhere(X_col<=thr).flatten(),np.argwhere(X_col>thr).flatten()

    def _information_gain(self,X_col,y,thr):
        parent_entropy = self._entropy(y)
        left_index , right_index = self._split(X_col,thr)
        if len(left_index) == 0 or len(right_index) == 0:
            return 0
        n = len(y)
        n_l,n_r = len(left_index),len(right_index)
        e_l,e_r = self._entropy(y[left_index]),self._entropy(y[right_index])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        return parent_entropy-child_entropy

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        if len(np.unique(y)) > 10:
            y_binned = pd.cut(y, bins=10, labels=False, duplicates='drop')
            y_binned = y_binned[~np.isnan(y_binned)]
            if len(y_binned) == 0:
                return 0
            y = y_binned.astype(int)
        else:
            y = y.astype(int)

        ps = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def predict(self,X):
        return np.array([self._traverse(self.root,x) for x in X])

    def _traverse(self,node,x):
        if node.is_leaf_node():
            return node.value
        if x[node.features]<=node.threshold:
            return self._traverse(node.left,x)
        return self._traverse(node.right,x)

clf = DecisionTree(max_depth=10)
clf.fit(X_train,Y_train)
prediction = clf.predict(X_test)
def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)
print(f"Accuracy : {accuracy(Y_test,prediction)}")