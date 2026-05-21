from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

X = data.data
Y = data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

clf = SVC(kernel='linear',C=3)
clf.fit(X_train,Y_train)

clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(X_train,Y_train)

print(f"KNN : {clf2.score(X_test,Y_test)}")
print(f"SVC : {clf.score(X_test, Y_test)}")
