import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Input,Dense,BatchNormalization,Dropout,LSTM

data = pd.read_csv("spam.csv")

text = [t for t in data['text']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)

vocab_size = len(tokenizer.word_index)+1

sequence = tokenizer.texts_to_sequences(text)
padded = pad_sequences(sequence,maxlen=max([len(s) for s in sequence]),padding='pre',truncating='post')

X = np.array(padded,dtype=np.int32)
Y = np.array(data['label'].map({'ham':0,'spam':1}))

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Embedding(vocab_size, 64))
model.add(LSTM(64))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train,Y_train,validation_data=(X_test,Y_test),verbose=1,epochs=5,batch_size=32)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()



def generate():
    text = input("Enter Your sms : ")

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=X_train.shape[1], padding='pre', truncating='post')

    padded = np.array(padded,dtype=np.int32)
    y = model.predict(padded)
    print('spam' if y[0][0]>0.5 else 'ham')

generate()
