import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

data = datasets.load_breast_cancer()

X = data.data
Y = data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)



def entropy(y):
    ps = np.bincount(y)/len(y)
    return -np.sum([p*np.log2(p) for p in ps if p>0])

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self,X,Y):
        #grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root = self._grow_tree(X,Y)

    def _grow_tree(self,X,Y,depth=0):
        n_samples,n_features = X.shape
        n_labels = len(np.unique(Y))

        #stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(Y)
            return Node(value = leaf_value)

        feat_idxs =  np.random.choice(n_features,self.n_feats,replace=False)

        #greedy search
        best_feat , best_thresh = self._best_criteria(X,Y,feat_idxs)
        left_indices , right_indices = self._split(X[:,best_feat],best_thresh)
        left = self._grow_tree(X[left_indices,:],Y[left_indices],depth+1)
        right = self._grow_tree(X[right_indices,:],Y[right_indices],depth+1)
        return Node(best_feat,best_thresh,left,right)

    def _best_criteria(self,X,Y,feat_idxs):
        best_grain = -1
        split_idxs , split_threh = None,None

        for feat_idx in feat_idxs:
            X_Column = X[:,feat_idx]
            thresholds = np.unique(X_Column)
            for thr in thresholds:
                gain = self._information_gain(Y,X_Column,thr)
                if gain > best_grain:
                    best_grain = gain
                    split_idxs = feat_idx
                    split_threh = thr

        return split_idxs,split_threh

    def _information_gain(self,Y,X_column,split_threh):
        #parent entropy
        parent_entropy = entropy(Y)

        #generate split
        left_idxs , right_idxs = self._split(X_column,split_threh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        #weighted average
        n = len(Y)
        n_l , n_r = len(left_idxs),len(right_idxs)
        e_l,e_r = entropy(Y[left_idxs]),entropy(Y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        return parent_entropy-child_entropy


    def _split(self,X_Column,split_threh):
        return np.argwhere(X_Column<=split_threh).flatten(),np.argwhere(X_Column>split_threh).flatten()

    def _most_common_label(self,Y):
        return Counter(Y).most_common(1)[0][0]

    def predict(self,X):
        #traverse tree
        return np.array([self._traverse(x,self.root) for x in X])

    def _traverse(self,x,node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x,node.left)
        return self._traverse(x,node.right)

clf = DecisionTree(max_depth=10)
clf.fit(X_train,Y_train)
prediction = clf.predict(X_test)

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

print(accuracy(Y_test,prediction))

