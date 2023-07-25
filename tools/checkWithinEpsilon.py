#check an image is within epsilon distance of original image
#get counter example
#get original image
#compare with assert_allclose

import re
import numpy as np
import onnxruntime as ort
import ast
import tensorflow as tf
from tensorflow import keras

path_to_log1 = "../logs/fashion1l32n/onelayer32n0.1-100.txt" #counterexample
path_to_log2 = "../logs/fashion1l32n/onelayer32n0.01-100.txt" #no counterexample

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0


def getCounterexample(content):
    #as string
    counterexample = re.sub(r'[\s\S]*?perturbation: ', '', content)

    #as list
    counterexample = ast.literal_eval(counterexample)

    return counterexample

def checkCounterexample(file, index, epsilon):
    f = open(file, "r")

    content = f.read()

    if "proved no counterexample exists" in content:
        print("Index: " + str(index) + " with epsilon: " + str(epsilon) + " no counterexample given")
    elif "counterexample found" in content:
        counterexample = getCounterexample(content)
        counterexample = np.array([np.array(row) for row in counterexample])
        #get original image
        original_image = test_images[index]

        #compare with allclose
        #np.testing.assert_allclose(image.numpy(), pgd_point.numpy(), rtol = 0, atol=0.05, verbose='True')
        #print("Within epsilon: ", np.allclose(counterexample, original_image, rtol=0, atol=0.051))
        #np.testing.assert_allclose(counterexample, original_image, rtol=0, atol=0.051)
        if np.allclose(counterexample, original_image, rtol=0, atol=0.05):
            print("Counterexample at index: " + str(index) + " with epsilon: " + str(epsilon) + " is within epsilon")
        else:
            print("Counterexample at index: " + str(index) + " with epsilon: " + str(epsilon) + " is not within epsilon")
        
    else: 
        print("error at: " + index)

    f.close()

checkCounterexample(path_to_log1, 100, 0.1)
checkCounterexample(path_to_log2, 100, 0.01)

print("-------------Start-------------------")

#0.01

for index in range(100,500):
    filepath = "../logs/fashion1l32n/onelayer32n0.01-" + str(index) + ".txt" 
    checkCounterexample(filepath, index, 0.01)


#0.05
for index in range(100,500):
    filepath = "../logs/fashion1l32n/onelayer32n0.05-" + str(index) + ".txt" 
    checkCounterexample(filepath, index, 0.05)

#0.1
for index in range(100,500):
    filepath = "../logs/fashion1l32n/onelayer32n0.1-" + str(index) + ".txt" 
    checkCounterexample(filepath, index, 0.1)

#0.5
for index in range(100,500):
    filepath = "../logs/fashion1l32n/onelayer32n0.5-" + str(index) + ".txt" 
    checkCounterexample(filepath, index, 0.5)