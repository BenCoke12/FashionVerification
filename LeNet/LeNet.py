import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

data = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

train_images = tf.pad(train_images, [[0, 0], [2,2], [2,2]])/255
test_images = tf.pad(test_images, [[0, 0], [2,2], [2,2]])/255
print(train_images.shape)
print(train_images[0])
print(train_labels[0])

train_images = tf.expand_dims(train_images, axis=3, name=None)
test_images = tf.expand_dims(test_images, axis=3, name=None)

print(train_images.shape)
print(test_images.shape)

val_images = train_images[-2000:,:,:,:]
val_labels = train_labels[-2000:]
train_images = train_images[:-2000,:,:,:]
train_labels = train_labels[:-2000]

model = keras.Sequential()
model.add(keras.layers.Conv2D(6, 5, activation='relu', input_shape=train_images.shape[1:]))
model.add(keras.layers.AveragePooling2D(2))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(16, 5, activation='relu'))
model.add(keras.layers.AveragePooling2D(2))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(120, 5, activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(84, activation='relu'))
model.add(keras.layers.Dense(10)) #, activation='relu'

print(model.summary())

model.compile(optimizer='Adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=128, epochs=10, validation_data=(val_images, val_labels))

print(model.evaluate(test_images, test_labels))

fig, axs = plt.subplots(2, 1, figsize=(12,12))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].legend(['Train', 'Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].legend(['Train', 'Val'])
plt.show()

model.save('./LeNet6')
