import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import random

data = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

#load model
model = tf.keras.models.load_model('../onnxNetworks/fashion1l')

def pgdAttack(epsilon, index, iterations):
    #load image
    image = tf.convert_to_tensor(test_images[index:index+1])

    #make one hot array for CategoricalCrossentropy
    label = [0]*10
    label[test_labels[index]] = 1
    label = tf.convert_to_tensor([label])

    #parameters
    alpha = 2/255

    #create starting point inside the epsilon ball of the original image
    pgd_point = []

    #append values within epsilon distance of the original point
    for i in image[0]:
        row = []
        for j in i:
            row.append(random.uniform((j - epsilon), (j + epsilon)))
        pgd_point.append(row)

    pgd_point = tf.expand_dims(tf.convert_to_tensor(pgd_point), 0)

    #check allclose
    #np.testing.assert_allclose(image.numpy(), pgd_point.numpy(), rtol = 0, atol=epsilon+0.0001, verbose='True')

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(pgd_point)
            prediction = model(pgd_point, training=False)
            loss = loss_object(label, prediction)

        gradient = tape.gradient(loss, pgd_point)

        signed_grad = tf.sign(gradient)

        adversarial_image = pgd_point + signed_grad * alpha

        pgd_point = tf.clip_by_value(adversarial_image, image[0] - epsilon, image[0] + epsilon)

    #np.testing.assert_allclose(pgd_point, image, rtol=0, atol=0.050001)
    if not(np.allclose(pgd_point, image, rtol=0, atol=(epsilon+0.0001))):
        print("bad at: " + str(epsilon) + ", " + str(index))

    #predicted_label = class_labels[np.argmax(model(pgd_point))]
    #true_label = class_labels[test_labels[index]]

    return pgd_point

def pgdAttackBySample(epsilon, sample_image, sample_label, iterations):
    #load image
    image = tf.convert_to_tensor(sample_image)

    #make one hot array for CategoricalCrossentropy
    label = [0]*10
    label[sample_label] = 1
    label = tf.convert_to_tensor([label])

    #parameters
    alpha = 2/255
    """
    #create starting point inside the epsilon ball of the original image
    pgd_point = []

    #append values within epsilon distance of the original point
    for i in image:
        row = []
        for j in i:
            row.append(random.uniform((j - epsilon), (j + epsilon)))
        pgd_point.append(row)

    pgd_point = tf.expand_dims(tf.convert_to_tensor(pgd_point), 0)
    """
    pgd_point = tf.expand_dims(image, 0)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(pgd_point)
            prediction = model(pgd_point, training=False)
            loss = loss_object(label, prediction)

        gradient = tape.gradient(loss, pgd_point)

        signed_grad = tf.sign(gradient)

        adversarial_image = pgd_point + signed_grad * alpha

        pgd_point = tf.clip_by_value(adversarial_image, image - epsilon, image + epsilon)


    #np.testing.assert_allclose(pgd_point, image, rtol=0, atol=0.050001)
    if not(np.allclose(pgd_point, image, rtol=0, atol=(epsilon+0.0001))):
        print("bad at: " + str(epsilon))

    return pgd_point

#pgd_point should be a random point inside eps ball
#perturbed image = pgd_point + (signed_grad * epsilon)
#clip back into epsilon tf.clip_by_value(pgd_point, )
#when training: how many pgd points generated per image

#call pgdAttack on 500 elements of the test set
#onnx runtime session
#run pgd_point through onnx onnxnetwork
#count whether same or different

def massAttack():
    #create record string
    results = "index,epsilon,outcome\n"

    #start runtime session
    ortSession = ort.InferenceSession("fashion1l.onnx")

    for epsilon in [0.01, 0.05, 0.1, 0.5]:
        for index in range(5):
            #perform attack
            pgd_point = pgdAttack(epsilon, index, 5)
            #sample_image = test_images[index]
            #sample_label = test_labels[index]
            #pgd_point = pgdAttackBySample(epsilon, sample_image, sample_label, 5)

            #run trough onnx network
            image = pgd_point.numpy().astype(np.float32)
            ort_inputs = {ortSession.get_inputs()[0].name: image}
            onnx_result = ortSession.run(None, ort_inputs)

            #true and predicted labels
            true_label = test_labels[index]
            predicted_label = np.argmax(onnx_result)

            #process Results and record outcome in string
            # 1 : attack successful
            # 0 : attack failed
            if true_label == predicted_label:
                results += str(index) + "," + str(epsilon) + ",0\n"
            elif true_label != predicted_label:
                results += str(index) + "," + str(epsilon) + ",1\n"
            else:
                print("bad at: " + str(index) + "," + str(epsilon))

    print(results)
    #store results in file
    file = open("pgdAttackResults.txt", "w")
    file.write(results)
    file.close()

def pgdTraining():
    #define model
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(keras.layers.Dense(10))

    #set parameters
    num_epochs = 5
    batch_size = 64

    #setup datasets for training
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    #optimiser and metrics
    optimizer = tf.keras.optimizers.Adam()
    scce_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    pgd_batch_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_metric = keras.metrics.SparseCategoricalAccuracy()

    #training epochs
    from datetime import datetime
    for epoch in range(num_epochs):
        print(f"\nStart of training epoch {epoch + 1}")
        counter = 0
        for batch_index, (x_batch, y_batch) in enumerate(train_dataset):
            #generate pgd samples for the batch
            pgd_samples = []

            for sample_image, sample_label in zip(x_batch, y_batch):
                pgd_point = pgdAttackBySample(0.02, sample_image, sample_label, 5)
                #pgd_point = tf.zeros([28,28])
                pgd_samples.append(tf.convert_to_tensor(pgd_point))

            #turn list into tensor for training
            pgd_samples = tf.convert_to_tensor(pgd_samples)
            pgd_samples = tf.squeeze(pgd_samples)

            counter += 1
            print("batch: " + str(counter) + "/1875")
            now = datetime.now()
            print(now.time())

            #do training
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                scce_loss_value = scce_batch_loss(y_batch, y_pred)

                y_pred_pgd = model(pgd_samples, training=True)
                pgd_loss_value = pgd_batch_loss(y_batch, y_pred_pgd)
                combined_loss = 0.5 * scce_loss_value + 0.5 * pgd_loss_value

            gradients = tape.gradient(combined_loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        train_acc = acc_metric.result()
        print(f"Accuracy over epoch: {train_acc}")
        acc_metric.reset_states()
        model.save(f'onnxNetworks/pgdTrained')

pgdTraining()
#massAttack()
