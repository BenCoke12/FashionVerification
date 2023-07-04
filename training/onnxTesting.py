import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tensorflow import keras
import onnx
import sys

#model location
filename = sys.argv[1]

#basic validation and sanity checks of the model - check onnx equiv to tf model
onnx_model = onnx.load("./onnxNetworks/" + filename + ".onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

image = test_images[1].astype(np.float32)
imageReshape = np.array([image])

ortSession = ort.InferenceSession("./onnxNetworks/" + filename + ".onnx")
ort_inputs = {ortSession.get_inputs()[0].name: imageReshape}
onnx_result = ortSession.run(None, ort_inputs)

print(onnx_result)
print(class_labels[np.argmax(onnx_result)])

tfModel = tf.keras.models.load_model('onnxNetworks/' + filename)

tf_result = tfModel(imageReshape)
print(tf_result)
print(class_labels[np.argmax(tf_result)])
print(type(tf_result.numpy()), type(onnx_result[0]))
np.testing.assert_allclose(tf_result.numpy(), onnx_result[0], rtol=1e-09, atol=1e-03)


#train and test accuracy
train_correct = 0
train_total = 0

for i in range(len(train_images)):
    image = train_images[i].astype(np.float32)
    imageReshape = np.array([image])
    ort_inputs = {ortSession.get_inputs()[0].name: imageReshape}
    onnx_result = ortSession.run(None, ort_inputs)
    if np.argmax(onnx_result) == train_labels[i]:
        train_correct += 1
    train_total += 1

print("Train Accuracy: ", round(train_correct/train_total, 3))
print("Correct: ", train_correct)
print("Total: ", train_total)

test_correct = 0
test_total = 0

for i in range(len(test_images)):
    image = test_images[i].astype(np.float32)
    imageReshape = np.array([image])
    ort_inputs = {ortSession.get_inputs()[0].name: imageReshape}
    onnx_result = ortSession.run(None, ort_inputs)
    if np.argmax(onnx_result) == test_labels[i]:
        test_correct += 1
    test_total += 1

print("Test Accuracy: ", round(test_correct/test_total, 3))
print("Correct: ", test_correct)
print("Total: ", test_total)
