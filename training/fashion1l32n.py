import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(10))

model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

trainEvaluation = model.evaluate(train_images, train_labels)
print("train evaluation", trainEvaluation)

testEvaluation = model.evaluate(test_images, test_labels)
print("test evaluation", testEvaluation)

prediction = model.predict(test_images[1:2])
print(prediction)
indexNumber = np.argmax(prediction, axis=1)
print(indexNumber)
item = class_labels[indexNumber[0]]
print(item)
print(class_labels[test_labels[1]])

plt.figure()
plt.imshow(test_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

model.save(f'onnxNetworks/fashion1l32n')
