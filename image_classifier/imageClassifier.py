import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras import layers, models
(training_images, training_labels), (testing_images, testing_labels) = tf.keras.datasets.cifar10.load_data()

training_images, testing_images = training_images / 255.0, testing_images / 255.0

class_names = ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]

for image in range(16):
    plt.subplot(4,4,image+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[image],cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[image][0]])

plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels))

loss,accuracy = model.evaluate(testing_images,testing_labels)
print(f"loss : {loss}")
print(f"Accuracy : {accuracy}")

model.save('image_classifier.keras')

# model = models.load_model('image_classifier.keras')


# img = cv.imread('horse.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# plt.imshow(img)

# prediction = model.predict(np.array([img]))  # Add batch dimension
# index = np.argmax(prediction)
# # print(prediction)
# print(f"Prediction is : {class_names[index]}")

# plt.show()
