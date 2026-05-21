import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X = data.data
Y = data.target
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

#logisticReggression
# class LogisticRegression:
#     def __init__(self,lr=0.001,epochs=1000):
#         self.weight=None
#         self.biases=None
#         self.lr=lr
#         self.epochs=epochs
#
#     def _sigmoid(self,l):
#         return 1/(1+(np.exp(np.clip(-l,-250,250))))
#
#     def fit(self,X,Y):
#         n_samples,n_features=X.shape
#         self.weight=np.zeros(n_features)
#         self.biases=0
#         for i in range(self.epochs):
#             linear_model = np.dot(X,self.weight)+self.biases
#             y_predicted = self._sigmoid(linear_model)
#             dw = (1/n_samples)*(np.dot(X.T,y_predicted-Y))
#             db = (1/n_samples)*np.sum(y_predicted-Y)
#             self.weight -= self.lr*dw
#             self.biases -= self.lr*db
#
#     def predict(self,X):
#         linear_model = np.dot(X, self.weight) + self.biases
#         y_predicted = self._sigmoid(linear_model)
#         return (y_predicted >= 0.5).astype(int)
#
# clf = LogisticRegression(lr=0.001,epochs=2000)
# clf.fit(X_train,Y_train)
# prediction = clf.predict(X_test)
# def accuracy(y_true,y_pred):
#     return np.sum(y_true==y_pred)/len(y_true)
# print(f"Accuracy : {accuracy(Y_test,prediction)}")


class SVM:
    def __init__(self,lamdaP=0.01,lr=0.001,epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.lamdaP = lamdaP
        self.w = None
        self.b = None

    def fit(self,X,Y):
        y_ = np.where(Y<=0 , -1 , 1)
        n_samples , n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.epochs):
            for idx , x_i in enumerate(X):
                condition = y_[idx]*(np.dot(x_i,self.w)+self.b)>=1
                if condition:
                    self.w -= (self.lr)*(2 * self.lamdaP*(self.w))
                    self.b -= 0
                else:
                    self.w -= (self.lr)*((2 * self.lamdaP * self.w)-np.dot(x_i,y_[idx]))
                    self.w -= (self.lr)*(y_[idx])

    def predict(self,X):
        linear_model = np.dot(X,self.w)-self.b
        return np.sign(linear_model)

clf = SVM(lr=0.001,epochs=1000)
clf.fit(X_train,Y_train)
prediction = clf.predict(X_test)
def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)
print(f"Accuracy : {accuracy(Y_test,prediction)}")
