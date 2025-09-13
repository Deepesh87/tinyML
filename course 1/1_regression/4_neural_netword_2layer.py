import numpy as np
import tensorflow as tf

# define a neural network with 2 layers
# for more information on TF functions see: https://www.tensorflow.org/api_docs

my_layer_1 = tf.keras.layers.Dense(units=2, input_shape=[1])

my_layer_2 = tf.keras.layers.Dense(units=1)

model = tf.keras.Sequential([my_layer_1, my_layer_2])

model.compile(optimizer='sgd', loss='mean_squared_error')



xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)

ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

 

model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
print(my_layer_1.get_weights())
print(my_layer_2.get_weights())
print(my_layer_1.get_weights()[0][0][0])  # weight of first neuron in first layer
print(my_layer_1.get_weights()[0][0][1])  # weight of second neuron in first layer
layer1_b1 = (my_layer_1.get_weights()[1][0]) # bias of first neuron in first layer
layer1_b2 = (my_layer_1.get_weights()[1][1]) # bias of second neuron in first layer
print(layer1_b1, layer1_b2)  # biases of first layer neurons
print(my_layer_2.get_weights()[0][0])  # weight of first neuron in second layer
print(my_layer_2.get_weights()[0][1])  # weight of second neuron in second layer
print(my_layer_2.get_weights()[1][0])  # bias of first neuron in second layer

