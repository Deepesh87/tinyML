import tensorflow as tf
import sys

data = tf.keras.datasets.mnist

(training_images, training_labels), (val_images, val_labels) = data.load_data()

# Normalize the images to a range of 0 to 1 so model can learn better
training_images = training_images / 255.0
val_images = val_images / 255.0

#reLU means Rectified Linear Unit, a common activation function in neural networks
# softmax is used in the output layer for multi-class classification problems
layer_1 = tf.keras.layers.Dense(20, activation=tf.nn.relu) # Hidden layer with 20 neurons and ReLU activation
layer_2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax) # Output layer with 10 neurons for each digit (0-9)
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)), # Flatten the 28x28(=784) images to a 1D array so the model can process them
                                    layer_1,
                                    layer_2])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, validation_data=(val_images, val_labels) ,epochs=20) #adding val so we see train and val performance in each epoch
model.evaluate(val_images, val_labels)

classifications = model.predict(val_images)
print(classifications[0])
print(val_labels[0])

#inspect the weights of the first layer
print(layer_1.get_weights()) # an array of weights and biases, number of weights is 20x784= 15680 (20 neurons, each with 784 inputs from the flattened image)
print(layer_1.get_weights()[0].size) # size of the weights array is 15680
print(layer_1.get_weights()[1].size) # size of the biases array is 20 (one bias for each neuron in the first layer)


#inspect the weights of the second layer
print(layer_2.get_weights()) # an array of weights and biases, number of weights is 10x20=200 (10 neurons in the output layer, each with 20 inputs from the first layer)
print(layer_2.get_weights()[0].size) # size of the weights array is 200
print(layer_2.get_weights()[1].size) #size of the biases array is 10 (one bias for each neuron in the second layer)









