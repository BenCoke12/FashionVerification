import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from vehicle_lang.compile import Target, to_python

# 1 layer of 32 nodes
# trained with vehicle loss function

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(10))

