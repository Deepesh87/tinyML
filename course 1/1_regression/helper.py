import numpy as np
import tensorflow as tf
import numpy as np

# 1 ====testing the loss function==================================
def loss(predicted_y, target_y):
    """
    implements the Mean Squared Error (MSE) loss function
    """
    return tf.reduce_mean(tf.square(predicted_y - target_y))

# NumPy arrays
predicted_y = np.array([2.5, 0.0, 2.1, 7.8]) # can also use tf.constant([2.5, 0.0, 2.1, 7.8], dtype=tf.float32)
target_y    = np.array([3.0, -0.5, 2.0, 7.5]) # can also use tf.constant([3.0, -0.5, 2.0, 7.5], dtype=tf.float32)

# Call the TensorFlow loss function
mse_loss = loss(predicted_y, target_y)
print("MSE Loss:", mse_loss.numpy())

# 2 ====testing the linear regression model========================

# Define our simple linear regression model
class Model(object):
  def __init__(self):
    # Initialize the weights
    self.w = tf.Variable(INITIAL_W)
    self.b = tf.Variable(INITIAL_B)

  def __call__(self, x):
    return self.w * x + self.b   # call method to make the class instance callable
  
INITIAL_W = 10.0
INITIAL_B = 10.0
LEARNING_RATE=0.09

model =Model()
# Test the model with some inputs
test_inputs = np.array([1.0, 2.0, 3.0]) # tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
predicted_outputs = model(test_inputs)
print(predicted_outputs.numpy())

# 3 ====testing the linear regression model========================