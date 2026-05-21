import numpy as np
from decisionTreeFromScratch import DecisionTree
from collections import Counter
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_breast_cancer

data = datasets.load_breast_cancer()

X = data.data
Y = data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)


def bootstrap_sample(X,Y):
    n_samples = X.shape[0]
    index = np.random.choice(n_samples,size=n_samples,replace=True)
    return X[index],Y[index]

def most_common_label(Y):
    return Counter(Y).most_common(1)[0][0]


class RandomForest:
    def __init__(self,n_trees=100,min_samples_split=2,max_depth=100,n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self,X,Y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,max_depth=self.max_depth,n_feats=self.n_feats)
            X_sample,Y_sample = bootstrap_sample(X,Y)
            tree.fit(X_sample,Y_sample)
            self.trees.append(tree)


    def predict(self,X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds,0,1)
        y_preds = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_preds)


clf = RandomForest(n_trees=10)
clf.fit(X_train,Y_train)

prediction = clf.predict(X_test)

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

print(accuracy(Y_test,prediction))