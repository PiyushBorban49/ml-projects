
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import datetime
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

data = pd.read_csv('database.csv')
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'],
                                 format='%m/%d/%Y %H:%M:%S',
                                 errors='coerce')
data['TimeStamp'] = data['DateTime'].astype('int64') // 10**9

X = data[['TimeStamp','Latitude','Longitude']]
y = data[['Depth','Magnitude']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = Sequential()
model.add(tf.keras.layers.Dense(64,activation='relu',input_shape=(3,)))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(2,activation='relu'))
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer = 'adam')
model.fit(X_train,y_train)
loss,accuracy = model.evaluate(X_test,y_test)
print(accuracy)
print(loss)