import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from vehicle_lang.compile import Target, to_python
from pathlib import Path
import random
from typing import Any, Dict, Iterator

# 1 layer of 32 nodes
# trained with vehicle loss function

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dense(10))

# Load and compile Vehicle specification
specification_filename = "fashionRobustTrain.vcl"
specification_path = Path(__file__).parent.parent / "vclspecs" / specification_filename

# Create function to apply model
def apply_model(input):
  return model(tf.expand_dims(input, axis=0), training=True)[0]

# What is a sampler, what is its purpose, when should they be used, fgsm/pgd?
def sampler_for_perturbation(_context: Dict[str, Any]) -> Iterator[float]:
    for _ in range(0, 10):
        yield random.uniform(0.5, 0.5)


robust = to_python(specification_path,
                   target=Target.LOSS_DL2,
                   samplers={"perturbation":sampler_for_perturbation})["robust"]

#need to hard code an epsilon?
epsilon = 0.01

#need to include n with the fashionRobustTrain script? can it be inferred or not?
#order of parameters matters - should match spec
def robustness_loss(images, labels, n):
    return robust(classifier=apply_model)(epsilon=epsilon)(n=n)(images=images)(labels=labels)


#try to vary batch size to 32, might not be possible with fashionRobustness.vcl
batch_size = 1

# Prepare training data
data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

#increase to 5 for actual run
num_epochs = 1

optimizer = tf.keras.optimizers.Adam()
scce_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

acc_metric = keras.metrics.SparseCategoricalAccuracy()

#tune weight of crossentropy vs robustness
scce_loss_weight = 0
robustness_weight = 1

#Training Loop
for epoch in range(num_epochs):
    print(f"\nStart of training epoch {epoch + 1}")
    for batch_index, (x_batch, y_batch) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            scce_loss = scce_batch_loss(y_batch, y_pred)
            robust_loss = robustness_loss(x_batch, y_batch, batch_size)
            print(robust_loss)
            weighted_loss = (scce_loss_weight * scce_loss + robustness_weight * robust_loss)

        gradients = tape.gradient(weighted_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    train_acc = acc_metric.result()
    print(f"Accuracy over epoch: {train_acc}")
    acc_metric.reset_states()

#Test loop
for batch_index, x_batch, y_batch in enumerate(test_dataset):
    y_pred = model(x_batch, training=True)
    acc_metric.update_state(y_batch, y_pred)

test_acc = acc_metric.result()
print(f"Accuracy over Test set: {test_acc}")
acc_metric.reset_states()