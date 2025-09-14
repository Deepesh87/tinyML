import numpy as np
import tensorflow as tf

# define a neural network with 2 layers
# for more information on TF functions see: https://www.tensorflow.org/api_docs

my_layer_1 = tf.keras.layers.Dense(units=2, input_shape=[1]) # input shape Only needed in the first layer.

my_layer_2 = tf.keras.layers.Dense(units=1) ## second hidden layer with 1 neurons

model = tf.keras.Sequential([my_layer_1, my_layer_2])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500, verbose=0)

print("prediction for x= 10.0 is: ",model.predict(np.array([10.0]))[0][0])
print("Hidden layer 1 weights are: ",my_layer_1.get_weights())
print("Hidden layer 2 weights are: ",my_layer_2.get_weights())
print("Weight of first neuron in Layer 1: ",my_layer_1.get_weights()[0][0][0])  # weight of first neuron in first layer
print("Weight of second neuron in Layer 1: ",my_layer_1.get_weights()[0][0][1])  # weight of second neuron in first layer

layer1_b1 = (my_layer_1.get_weights()[1][0]) # bias of first neuron in first layer
layer1_b2 = (my_layer_1.get_weights()[1][1]) # bias of second neuron in first layer
print("Biases of First Layer Neurons are: ",layer1_b1, layer1_b2)  # biases of first layer neurons
print("Weight of first neuron in Layer 2:  ",my_layer_2.get_weights()[0][0])  # weight of first neuron in second layer
print("Weight of first neuron in Layer 2:  ",my_layer_2.get_weights()[0][1])  # weight of second neuron in second layer
print("Biases of the only Neurons in Layer 2: ",my_layer_2.get_weights()[1][0])  # bias of first neuron in second layer

