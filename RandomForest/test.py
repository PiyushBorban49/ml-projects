import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from win32gui import DestroyIcon

data = load_iris()

X = data.data
Y = data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

def entropy(Y):
    ps = np.bincount(Y)/len(Y)
    return -np.sum([p*np.log2(p) for p in ps if p>0])

class Node:
    def __init__(self,features=None,threshold=None,left=None,right=None,*,value=None):
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self,X,Y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,Y)

    def _grow_tree(self,X,Y,depth=0):
        n_samples,n_features = X.shape
        n_labels = len(np.unique(Y))

        if depth >= self.max_depth or n_labels == 1 or n_samples<self.min_samples_split:
            return Node(value=self._most_common_label(Y))

        feats_index = np.random.choice(n_features,self.n_features,replace=False)

        best_features,best_threshold = self._best_criteria(Y,X,feats_index)
        left_index,right_index = self._spilt(X[:,best_features],best_threshold)
        left = self._grow_tree(X[left_index,:],Y[left_index],depth+1)
        right = self._grow_tree(X[right_index,:],Y[right_index],depth+1)
        return Node(best_features,best_threshold,left,right)


    def _best_criteria(self,Y,X,feat_index):
        best_gain = -1
        split_features,split_threshold = None,None

        for feats in feat_index:
            X_col = X[:,feats]
            threshold = np.unique(X_col)
            for thr in threshold:
                gain = self._information_gain(Y,X_col,thr)
                if best_gain<gain:
                    best_gain = gain
                    split_features = feats
                    split_threshold = thr

        return split_features,split_threshold

    def _information_gain(self,Y,X_col,thr):
        parent_entropy = entropy(Y)
        left_index,right_index = self._spilt(X_col,thr)

        if len(left_index) == 0 or len(right_index) == 0:
            return 0

        n = len(Y)
        n_left,n_right = len(left_index),len(right_index)
        e_left,e_right = entropy(Y[left_index]),entropy(Y[right_index])
        child_entropy = (n_left/n)*e_left + (n_right/n)*e_right
        return parent_entropy-child_entropy


    def _spilt(self,X_col,thr):
        return np.argwhere(X_col<=thr).flatten(),np.argwhere(X_col>thr).flatten()

    def _most_common_label(self,Y):
        return Counter(Y).most_common(1)[0][0]

    def predict(self,X):
        return np.array([self._traverse(x,self.root) for x in X])

    def _traverse(self,x,node):
        if node.is_leaf_node():
            return node.value

        if x[node.features]<=node.threshold:
            return self._traverse(x,node.left)
        return self._traverse(x,node.right)

clf = DecisionTree(max_depth=10)
clf.fit(X_train,Y_train)
prediction = clf.predict(X_test)

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

print(accuracy(Y_test,prediction))
