import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

exercise = pd.read_csv("exercise.csv")
calories = pd.read_csv("calories.csv")

exercise['Calories'] = calories['Calories']
exercise['Gender'] = np.where(exercise['Gender'] == "male", 1, 0)
exercise = exercise.drop(['User_ID'],axis=1)

X = exercise.drop(['Calories'],axis=1)
Y = exercise['Calories']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.2)

clf = XGBRegressor()
clf.fit(X_train,Y_train)
print("Accuracy : ",clf.score(X_test,Y_test))