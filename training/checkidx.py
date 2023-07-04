import matplotlib.pyplot as plt
import numpy as np
import idx2numpy
import tensorflow as tf
from tensorflow import keras

#images = idx2numpy.convert_from_file('./data/50-99Images.idx')
#labels = idx2numpy.convert_from_file('./data/50-99Labels.idx')

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

images = idx2numpy.convert_from_file('./data/individuals/Image0.idx')
labels = idx2numpy.convert_from_file('./data/individuals/Label0.idx')

index = 0
print("length of images array: ", len(images))
print("length of labels: ", len(labels))
print(images[index])
print(labels[index])
print(test_labels[0])

class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(class_labels[labels[index]])


plt.figure()
plt.imshow(images[index])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

images = idx2numpy.convert_from_file('./data/individuals/Image99.idx')
labels = idx2numpy.convert_from_file('./data/individuals/Label99.idx')

index = 0
print("length of images array: ", len(images))
print("length of labels: ", len(labels))
print(images[index])
print(labels[index])
print(test_labels[99])

class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(class_labels[labels[index]])


plt.figure()
plt.imshow(images[index])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure()
plt.imshow(test_images[99])
plt.colorbar()
plt.grid(False)
plt.show()
