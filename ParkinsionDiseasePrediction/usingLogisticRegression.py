import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("parkinson_disease.csv")
X = data.drop([
    'locPctJitter',
    'locAbsJitter',
    'rapJitter',
    'ppq5Jitter',
    'ddpJitter',
    'locShimmer',
    'locDbShimmer',
    'apq3Shimmer',
    'apq5Shimmer',
    'apq11Shimmer',
    'ddaShimmer'],axis=1).values
Y = data['class'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)


# Learn Advanced Data Handeling

class LogisticRegression():
    def __init__(self,epochs=20000,learing_rate=0.01):
        self.learning_rate = learing_rate
        self.epochs = epochs
        self.weights = None
        self.biases = None

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def fit(self,X,Y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.biases = 0
        for _ in range(self.epochs):
            linear_model = np.dot(X,self.weights)+self.biases
            y_prediction = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.sum(np.dot(X.T, y_prediction - Y))
            db = (1 / n_samples) * np.sum(y_prediction - Y)
            self.weights -= self.learning_rate * dw
            self.biases -= self.learning_rate * db

    def predict(self,X):
        linear_model = np.dot(X, self.weights) + self.biases
        y_prediction = self._sigmoid(linear_model)
        return [1 if i>0.5 else 0 for i in y_prediction]


clf = LogisticRegression(epochs=20000,learing_rate=0.01)
clf.fit(X,Y)
def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)
print(accuracy(Y,clf.predict(X)))
