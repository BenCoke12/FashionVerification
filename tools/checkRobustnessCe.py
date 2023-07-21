#go through logs and check whether a log contains a counter example or not
#get the counterexample if it does exist
#get the true label for the image
#run the counterexample through the network
#get the predicted label for the image
#compare predicted label to the true label
#if they are different then this is a counterexample
#if they are the same then it is not
#loop over 100-499 files
#loop over each epsilon; 0.01, 0.05, 0.1, 0.5

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

#start ort session
ortSession = ort.InferenceSession("../onnxnetworks/fashion1l32n.onnx")


def getCounterexample(content):
    #as string
    counterexample = re.sub(r'[\s\S]*?perturbation: ', '', content)

    #as list
    counterexample = ast.literal_eval(counterexample)

    return counterexample

def getPrediction(counterexample):
    image = np.array([np.array(row) for row in counterexample])
    imageForRuntime = image.astype(np.float32)
    imageReshape = np.array([imageForRuntime])
    ort_inputs = {ortSession.get_inputs()[0].name: imageReshape}
    prediction = ortSession.run(None, ort_inputs)
    return prediction


def checkCounterexample(file, index, epsilon):
    f = open(file, "r")

    content = f.read()

    if "proved no counterexample exists" in content:
        print("Index: " + str(index) + " with epsilon: " + str(epsilon) + " no counterexample given")
    elif "counterexample found" in content:
        counterexample = getCounterexample(content)
        prediction = getPrediction(counterexample)
        predictedLabel = np.argmax(prediction)
        trueLabel = test_labels[index]
        #print("True label: ", trueLabel)
        #print("Predicted label: ", predictedLabel)
        if trueLabel == predictedLabel:
            print("Index: " + str(index) + " with epsilon: " + str(epsilon) + " wrongly declared as counterexample")
        elif trueLabel != predictedLabel:
            print("Index: " + str(index) + " with epsilon: " + str(epsilon) + " correctly identified as counterexample")
    else: 
        print("error at i")

    f.close()

checkCounterexample(path_to_log1, 100, 0.1)
checkCounterexample(path_to_log2, 100, 0.01)

print("-------------Start-------------------")

#0.01
'''
for index in range(100,500):
    filepath = "../logs/fashion1l32n/onelayer32n0.01-" + str(index) + ".txt" 
    checkCounterexample(filepath, index, 0.01)
'''

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