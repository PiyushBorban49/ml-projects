import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("Rainfall.csv")
data = data.dropna(subset=["day","pressure" ,"maxtemp","temparature","mintemp","dewpoint","humidity" ,"cloud" ,"rainfall","sunshine","winddirection","windspeed"])
X = data[["pressure" ,"maxtemp","temparature","mintemp","dewpoint","humidity" ,"cloud","sunshine","winddirection","windspeed"]].values
Y = data["rainfall"].values
le = LabelEncoder()
Y = le.fit_transform(Y)

ros = RandomOverSampler()
X,Y = ros.fit_resample(X,Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)



def entropy(Y):
    if len(Y) == 0:
        return 0
    Y = Y.astype(int)
    ps = np.bincount(Y)/len(Y)
    return -np.sum([p*np.log2(p) for p in ps if p>0])

class Node:
    def __init__(self,features=None,threshold=None,left=None,right=None,*,value=None):
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_Leaf_Node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_features=None,max_bins=20):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.max_bins = max_bins

    def fit(self,X,Y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features,X.shape[1])
        self.root = self._grow_tree(X,Y)

    def _most_common_labels(self,Y):
        if len(Y) == 0:
            return 0

        return Counter(Y).most_common(1)[0][0]

    def _grow_tree(self,X,Y,depth=0):
        n_samples,n_features=X.shape
        n_labels = len(Y)

        if depth>=self.max_depth or n_labels == 1 or n_samples<self.min_samples_split:
            return Node(value = self._most_common_labels(Y))

        feats_index = np.random.choice(n_features,self.n_features,replace=False)

        best_features,best_threshold = self._best_criteria(X,Y,feats_index)
        if best_features is None:
            return Node(value=self._most_common_labels(Y))
        left_index,right_index = self._split(X[:,best_features],best_threshold)
        if len(left_index) == 0 or len(right_index) == 0:
            return Node(value=self._most_common_labels(Y))
        left = self._grow_tree(X[left_index,:],Y[left_index],depth+1)
        right = self._grow_tree(X[right_index,:],Y[right_index],depth+1)
        return Node(best_features,best_threshold,left,right)

    def _best_criteria(self,X,Y,feats_index):
        best_gain = -1
        split_features,split_threshold = None,None
        for feats in feats_index:
            X_col = X[:,feats]
            if len(np.unique(X_col)) > self.max_bins:
                threshold = np.percentile(X_col, np.linspace(0, 100, self.max_bins))
            else:
                threshold = np.unique(X_col)
            for thre in threshold:
                gain = self._information_gain(X_col,Y,thre)
                if gain > best_gain:
                    best_gain = gain
                    split_threshold = thre
                    split_features = feats
        return split_features,split_threshold

    def _split(self,X_col,thr):
        return np.argwhere(X_col<=thr).flatten(),np.argwhere(X_col>thr).flatten()

    def _information_gain(self,X_col,Y,thr):
        parent_entropy = entropy(Y)

        left_index,right_index = self._split(X_col,thr)
        if len(left_index)==0 or len(right_index)==0:
            return 0

        n = len(Y)
        n_l,n_r = len(left_index),len(right_index)
        e_l,e_r = entropy(Y[left_index]),entropy(Y[right_index])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        return parent_entropy-child_entropy

    def predict(self,X):
        return np.array([self._traverse(self.root,x) for x in X])

    def _traverse(self,node,x):
        if node.is_Leaf_Node():
            return node.value

        if x[node.features]<=node.threshold:
            return self._traverse(node.left,x)
        return self._traverse(node.right,x)

clf = DecisionTree(max_depth=10)
clf.fit(X_train,Y_train)
prediction = clf.predict(X_test)
def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)
print(f"Accuarcy : {accuracy(Y_test,prediction)}")

