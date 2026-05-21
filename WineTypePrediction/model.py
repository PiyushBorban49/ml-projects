import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
from keras import regularizers
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers,models

red = pd.read_csv("winequality-red.csv",sep=";")
white = pd.read_csv("winequality-white.csv",sep=";")
red['type']=0
white['type']=1
data = pd.concat([red,white])
X = data[["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]].values
Y = data['type'].values

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# 12->128->128->128->2
model = Sequential()
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu, input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.01)))
model.add(tf.keras.layers.Dense(units=2, activation=tf.nn.softmax))
model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
history = model.fit(X_train,Y_train,epochs=25,verbose=1,validation_split=0.2)
loss,accuracy = model.evaluate(X_test,Y_test)
print(loss)
print(accuracy)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()