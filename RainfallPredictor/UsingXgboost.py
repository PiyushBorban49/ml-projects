import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sympy import subsets
from xgboost import XGBRegressor, XGBClassifier

# Imported data from weatherAUS.csv
data = pd.read_csv('weatherAUS.csv')

# Replacing Values of yes and no in data
data.replace("Yes",1,inplace=True)
data.replace("No",0,inplace=True)

# Created new column called TimeStamp
data['TimeStamp'] = pd.to_datetime(data['Date']).apply(lambda x : x.timestamp())

# Using SimpleImputer to fill null values
null_cols = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday']
for i in null_cols:
    data[i] = SimpleImputer().fit_transform(data[i].values.reshape(-1,1))

data['CountryID'] = pd.factorize(data.Location)[0]

# Preparing training and testing data
X = data[null_cols+['CountryID','TimeStamp']].values.astype(int)
Y = data['RainTomorrow'].values

# Splitting data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

# Using StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Using Xgboost
clf = XGBClassifier()
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))