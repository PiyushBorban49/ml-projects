import os
import tensorflow.python.keras.models
from keras.src.metrics.accuracy_metrics import accuracy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras import layers

mnist = tf.keras.datasets.mnist
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(10,activation='softmax'))
#
#
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#
# model.fit(X_train,Y_train,epochs=10)
#
# model.save('handwritten.keras')


flag,axes = plt.subplots(2,5,figsize=(12,5))
axes = axes.flatten()

for i in range(10):
    img_index = np.random.randint(1,len(X_train))

    axes[i].imshow(X_train[img_index],cmap='gray')
    axes[i].set_title(f"Label:{Y_train[img_index]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

model = tf.keras.models.load_model('handwritten.keras')

loss,accuracy = model.evaluate(X_test,Y_test)

print(f"Loss : {loss}")
print(f"accuracy : {accuracy}")