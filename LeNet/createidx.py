import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import idx2numpy

data = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = tf.pad(train_images, [[0, 0], [2,2], [2,2]])/255
test_images = tf.pad(test_images, [[0, 0], [2,2], [2,2]])/255

print(test_images[0])
print(test_labels[0])

#plt.figure()
#plt.imshow(test_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()
#print(type(test_images[:1].numpy()))
#print(type(test_labels[:1]))
idx2numpy.convert_to_file('LeNet1Image.idx', test_images[:1].numpy())
idx2numpy.convert_to_file('LeNet1Label.idx', test_labels[:1])
