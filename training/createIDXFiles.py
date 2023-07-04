import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import idx2numpy

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

#idx2numpy.convert_to_file('./data/redDimImages.idx', test_images)
#idx2numpy.convert_to_file('./data/redDimLabels.idx', test_labels)

#set to 100
for i in range(100):
    print(i)
    idx2numpy.convert_to_file('./data/individuals/Image' + str(i) + '.idx', test_images[i:i+1])
    idx2numpy.convert_to_file('./data/individuals/Label' + str(i) + '.idx', test_labels[i:i+1])
