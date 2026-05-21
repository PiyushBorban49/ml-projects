import os
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras import Sequential
from keras.layers import Dense,SimpleRNN,Embedding,Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Dropout, BatchNormalization
from keras import regularizers

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=25000)

max_length = 250

X_train = np.array(X_train, dtype=object)
X_test = np.array(X_test, dtype=object)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

model = Sequential()
model.add(Embedding(25000,128,input_length=max_length))
model.add(BatchNormalization())
model.add(SimpleRNN(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,Y_train,epochs=5,batch_size=128,validation_data=(X_test,Y_test))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()