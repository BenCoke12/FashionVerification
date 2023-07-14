import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from tensorflow import keras
import onnx
import sys
import idx2numpy

#model location
filename = sys.argv[1]

#basic validation and sanity checks of the model - check onnx equiv to tf model
onnx_model = onnx.load("../onnxnetworks/" + filename + ".onnx")
onnx.checker.check_model(onnx_model)
print(onnx.helper.printable_graph(onnx_model.graph))

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

#counterexample =
image = idx2numpy.convert_from_file('../idxdata/individuals/Image30.idx')[0].astype(np.float32)
label = idx2numpy.convert_from_file('../idxdata/individuals/Label30.idx')
print(class_labels[label[0]])

plt.figure()
plt.imshow(image)
plt.colorbar()
plt.grid(False)
plt.show()

#image = test_images[29].astype(np.float32)
imageReshape = np.array([image])

ortSession = ort.InferenceSession("../onnxnetworks/" + filename + ".onnx")
ort_inputs = {ortSession.get_inputs()[0].name: imageReshape}
onnx_result = ortSession.run(None, ort_inputs)

print(onnx_result[0][0])
print(class_labels[np.argmax(onnx_result)])
"""
for i in range(500):
    image = test_images[i].astype(np.float32)
    imageReshape = np.array([image])
    ort_inputs = {ortSession.get_inputs()[0].name: imageReshape}
    onnx_result = ortSession.run(None, ort_inputs)
    if np.argmax(onnx_result) == 7:
        #print("Test image: ", i)
        #print("Label: ", test_labels[i])
        #print("Results: ", onnx_result)
        if onnx_result[0][0][5] > onnx_result[0][0][2]:
            print(i)
        else:
            pass
    else:
        pass
    """