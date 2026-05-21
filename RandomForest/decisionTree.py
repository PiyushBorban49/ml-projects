# information_gain = entropy(parent)-entropy(children)*[weighted average]
# entropy = summation(p(x).log(p(x))
# p(x) = no of that class appered in this node/total no of classes in this node

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


data = load_breast_cancer()

X = data.data
Y = data.target

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

clf = SVC(kernel="linear",C=3)
clf.fit(X_train,Y_train)


clf2 = DecisionTreeClassifier()
clf2.fit(X_train,Y_train)


clf3 = RandomForestClassifier()
clf3.fit(X_train,Y_train)

clf4 = KNeighborsClassifier()
clf4.fit(X_test,Y_test)


print(f"Support Vector Classification : {clf.score(X_test,Y_test)}")
print(f"Decision Tree Classification : {clf2.score(X_test,Y_test)}")
print(f"Random Forest Classification : {clf3.score(X_test,Y_test)}")
print(f"Kth Nearest Neighbour : {clf4.score(X_test,Y_test)}")