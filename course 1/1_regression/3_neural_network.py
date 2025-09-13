import numpy as np
import tensorflow as tf

# define a neural network with one neuron
# for more information on TF functions see: https://www.tensorflow.org/api_docs

my_layer = tf.keras.layers.Dense(units=1, input_shape=[1]) 

model = tf.keras.Sequential([my_layer])

# use stochastic gradient descent for optimization and
# the mean squared error loss function
model.compile(optimizer='sgd', loss='mean_squared_error') #stocastic gradient descent

# define some training data (xs as inputs and ys as outputs)
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# fit the model to the data (aka train the model)
model.fit(xs, ys, epochs=500)

print(model.predict(np.array([10.0])))
print(my_layer.get_weights())