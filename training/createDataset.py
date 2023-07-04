import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import idx2numpy

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

print(test_images[0])
print(test_labels[0])

idx2numpy.convert_to_file('./data/1Image.idx', test_images[:1])
idx2numpy.convert_to_file('./data/1Label.idx', test_labels[:1])
