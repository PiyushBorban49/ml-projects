import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("framingham.csv")
data = data.dropna(subset=["male","age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose","TenYearCHD"])
X = data[["male","age","education","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]].values
Y = data["TenYearCHD"].values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)


def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

class LogisticRegression:
    def __init__(self,learning_rate=0.001,epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.biases = None

    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.biases = 0

        for _ in range(self.epochs):
            linearmodel = np.dot(X, self.weights) + self.biases
            y_predicted = self._sigmoid(linearmodel)
            dw = (1 / n_samples) * np.sum(np.dot(X.T, y_predicted - Y))
            db = (1 / n_samples) * np.sum(y_predicted - Y)

            self.weights -= self.lr * dw
            self.biases -= self.lr * db

    def predict(self,X):
        linearmodel = np.dot(X,self.weights)+self.biases
        predicted = self._sigmoid(linearmodel)
        y_predicted = [np.int_(i) for i in predicted]
        return y_predicted

clf = LogisticRegression(learning_rate=0.001,epochs=2000)
clf.fit(X_train,Y_train)
prediction = clf.predict(X_test)
print(f"Accuracy : {accuracy(Y_test,prediction)}")


