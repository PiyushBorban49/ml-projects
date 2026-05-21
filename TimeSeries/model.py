import xgboost
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

data = pd.read_csv('AEP_hourly.csv')
data = data.set_index('Datetime')
data.index = pd.to_datetime(data.index)

data['hour'] = data.index.hour
data['dayofweek'] = data.index.dayofweek
data['quarter'] = data.index.quarter
data['month'] = data.index.month
data['year'] = data.index.year
data['dayofyear'] = data.index.dayofyear

train = data.loc[data.index<'01-01-2015']
test = data.loc[data.index>='01-01-2015']

Features = ['hour','dayofweek','quarter','month','year','dayofyear']
Target = 'AEP_MW'

X_train = train[Features]
Y_train = train[Target]

X_test = test[Features]
Y_test = test[Target]

reg = xgboost.XGBRegressor(n_estimators=1000,early_stopping_rounds=50,learning_rate=0.01)
reg.fit(X_train,Y_train,eval_set=[(X_train,Y_train),(X_test,Y_test)],verbose=100)

f1 = pd.DataFrame(data=reg.feature_importances_,index = reg.feature_names_in_,columns=['importance'])
f1.sort_values('importance').plot(kind='barh',title='Feature Importance')

test['prediction'] = reg.predict(X_test)
data = data.merge(test[['prediction']],how='left',left_index=True,right_index=True)
ax = data[['AEP_MW']].plot(figsize=(15,5))
data['prediction'].plot(ax = ax , style='.')
plt.legend('Truth Data','Prediction')
ax.set_title('Raw Dar and Prediction')

# data.plot(style='.',figsize=(15,5),color = 'red',title='Time Series')
plt.show()