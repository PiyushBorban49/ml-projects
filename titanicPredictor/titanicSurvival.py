import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

data = pd.read_csv("titanic_dataset.csv")
data = data.dropna(subset=["PassengerId","Survived","Pclass","Name","Sex","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"])

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]].values

Y = data.Survived.values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

def entropy(Y):
    ps = np.bincount(Y)/len(Y)
    return -np.sum([p*np.log2(p) for p in ps if p>0])

class Node:
    def __init__(self,features=None,threshold=None,left=None,right=None,*,value=None):
        self.value = value
        self.left = left
        self.right = right
        self.features = features
        self.threshold = threshold

    def isLeaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.root = None
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self,X,Y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,Y)

    def _grow_tree(self,X,Y,depth=0):
        n_samples,n_features = X.shape
        n_labels = len(np.unique(Y))

        if depth>=self.max_depth or n_labels == 1 or n_samples<self.min_samples_split:
            return Node(value=self._most_common_label(Y))

        feats_index = np.random.choice(n_features,self.n_features,replace=False)

        best_feature,best_threshold = self._best_criteria(Y,X,feats_index)
        left_index,right_index = self._split(X[:,best_feature],best_threshold)
        left = self._grow_tree(X[left_index,:],Y[left_index],depth+1)
        right = self._grow_tree(X[right_index,:],Y[right_index],depth+1)
        return Node(best_feature,best_threshold,left,right)

    def _best_criteria(self,Y,X,feats_index):
        best_gain = -1
        split_features , split_threshold = None,None
        for feats in feats_index:
            X_col = X[:,feats]
            threshold = np.unique(X_col)
            for thr in threshold:
                gain = self._information_gain(Y,X_col,thr)
                if gain > best_gain:
                    best_gain = gain
                    split_threshold = thr
                    split_features = feats
        return split_features,split_threshold

    def _information_gain(self,Y,X_col,thr):
        parent_entropy = entropy(Y)

        left_index,right_index = self._split(X_col,thr)

        if len(left_index) == 0 or len(right_index) == 0:
            return 0

        n = len(Y)
        n_l , n_r = len(left_index),len(right_index)
        e_l , e_r = entropy(Y[left_index]),entropy(Y[right_index])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r
        return parent_entropy-child_entropy

    def _split(self,X_col,thr):
        return np.argwhere(X_col<=thr).flatten(),np.argwhere(X_col>thr).flatten()

    def _most_common_label(self,Y):
        return Counter(Y).most_common(1)[0][0]

    def predict(self,X):
        return np.array([self._traverse(x,self.root) for x in X])

    def _traverse(self,x,node):
        if node.isLeaf():
            return node.value

        if x[node.features]<=node.threshold:
            return self._traverse(x,node.left)
        return self._traverse(x,node.right)


def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

clf = DecisionTree(max_depth=100)
clf.fit(X_train,Y_train)
prediction = clf.predict(X_test)
print(accuracy(Y_test,prediction))

