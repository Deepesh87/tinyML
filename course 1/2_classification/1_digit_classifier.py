import tensorflow as tf

data = tf.keras.datasets.mnist 

(training_images, training_labels), (val_images, val_labels) = data.load_data()

# Normalize the images to a range of 0 to 1 so model can learn better
training_images = training_images / 255.0
val_images = val_images / 255.0


# softmax is used in the output layer for multi-class classification problems
flatten_layer = tf.keras.layers.Flatten(input_shape=(28,28)) 
# Flatten the 28x28(=784) images to a 1D array so the model can process them

layer_1 = tf.keras.layers.Dense(20, activation=tf.nn.relu) # Hidden layer with 20 neurons and ReLU activation
# Each of the 784 input features connects to each of the 20 neurons.

layer_2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax) 
# Output layer with 10 neurons for each digit (0-9)


model = tf.keras.models.Sequential([flatten_layer, 
                                    layer_1,
                                    layer_2])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, validation_data=(val_images, val_labels) ,epochs=20) #adding val so we see train and val performance in each epoch

# Evaluate model on validation set
loss, accuracy = model.evaluate(val_images, val_labels, verbose=0)

print(f"Final Validation Loss: {loss:.4f}")
print(f"Final Validation Accuracy: {accuracy * 100:.2f}%")

classifications = model.predict(val_images)
# has the shape (10000, 10) for each of the 10k validation images, the model outputs an array of 10 probabilities (one for each digit 0-9)

print(classifications[0]) # print the array of 10 probabilities for the first validation image
print(val_labels[0]) # print the true label of the first validation image

#inspect the weights of the first layer
print(layer_1.get_weights()) # an array of weights and biases, number of weights is 20x784= 15680 (20 neurons, each with 784 inputs from the flattened image)
print(layer_1.get_weights()[0].size) # size of the weights array is 15680
print(layer_1.get_weights()[1].size) # size of the biases array is 20 (one bias for each neuron in the first layer)


#inspect the weights of the second layer
print(layer_2.get_weights()) # an array of weights and biases, number of weights is 10x20=200 (10 neurons in the output layer, each with 20 inputs from the first layer)
print(layer_2.get_weights()[0].size) # size of the weights array is 200
print(layer_2.get_weights()[1].size) #size of the biases array is 10 (one bias for each neuron in the second layer)



