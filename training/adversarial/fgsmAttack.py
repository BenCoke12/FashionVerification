import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

#load model
model = tf.keras.models.load_model('../onnxnetworks/fashion1l')

#load image
image = tf.convert_to_tensor(test_images[0:1])

label = [0]*10
label[test_labels[0]] = 1
label = tf.convert_to_tensor([label])

prediction = model.predict(image)
print(prediction)

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad

perturbations = create_adversarial_pattern(image, label)

plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]
plt.show()

def display_images(image, epsilon):
  #_, label, confidence = np.argmax(model.predict(image), axis=1)
  label = np.argmax(model.predict(image), axis=1)
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  #plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence*100))
  plt.title('epsilon: ' + str(epsilon))

  plt.savefig('perturbed'+str(epsilon)+'.png')

epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
  adv_x = image + eps*perturbations
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  display_images(adv_x, eps)

  print(class_labels[np.argmax(model.predict(adv_x))])
