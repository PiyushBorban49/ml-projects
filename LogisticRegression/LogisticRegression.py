import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
Y = data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

class LogisticRegression:
    def __init__(self,learning_rate = 0.001,epochs=1000):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.biases = None

    def _sigmoid(self,linearmodel):
        return 1/(1+np.exp(-linearmodel))

    def fit(self,X,Y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.biases = 0

        for _ in range(self.epochs):
            linearmodel = np.dot(X,self.weights)+self.biases
            y_predicted = self._sigmoid(linearmodel)
            dw = (1/n_samples) * np.sum(np.dot(X.T,y_predicted-Y))
            db = (1/n_samples) * np.sum(y_predicted-Y)

            self.weights -= self.learning_rate*dw
            self.biases -= self.learning_rate*db

    def predict(self,X):
        linearmodel = np.dot(X, self.weights) + self.biases
        y_predicted = self._sigmoid(linearmodel)
        y_predicted_class = [1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_class

logisticRegression = LogisticRegression(learning_rate=0.1,epochs=10000)
logisticRegression.fit(X_train,Y_train)
prediction = logisticRegression.predict(X_test)
print(f"Accuracy : {accuracy(Y_test,prediction)}")