import numpy as np
import tensorflow as tf
import numpy as np

# 1 ====testing the loss function==================================
print("Helper section 1: Loss Function")
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
print("Helper section 2: Linear Regression Model")

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
model =Model()
# Test the model with some inputs
test_inputs = np.array([1.0, 2.0, 3.0]) # tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
predicted_outputs = model(test_inputs)
print(predicted_outputs.numpy())


# 3 ====Gradient Descent========================
print("Helper section 3: Gradient Descent")
# manually adding init_w and init_b so iterations start from INITIAL_w and INITIAL_b

class Model(object):
    def __init__(self, w_init, b_init):
        self.w = tf.Variable(w_init, dtype=tf.float32)
        self.b = tf.Variable(b_init, dtype=tf.float32)
    def __call__(self, x):
        return self.w * x + self.b

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dw, db = t.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)
    return current_loss

INITIAL_W = 10.0
INITIAL_B = 10.0
LEARNING_RATE = 0.09

model_2 = Model(INITIAL_W, INITIAL_B)

xs = tf.constant([-1., 0., 1., 2., 3., 4.], dtype=tf.float32)
ys = tf.constant([-3., -1., 1., 3., 5., 7.], dtype=tf.float32)

print(f"Init: W={model_2.w.numpy():.2f}, b={model_2.b.numpy():.2f}")
for epoch in range(50):
    current_loss = train(model_2, xs, ys, learning_rate=LEARNING_RATE)
    print("Epoch %2d: W=%1.2f, b=%1.2f, loss=%2.5f" %
          (epoch, model_2.w.numpy(), model_2.b.numpy(), current_loss.numpy()))
    
#lets understand how these values came
# Init: W=10.00, b=10.00
# Epoch  0: W=-0.41, b=5.86, loss=715.66669

#==================manual calculation of first epoch========================
#y = wx + b
#y = 10x + 10
#xs = [-1., 0., 1., 2., 3., 4.]
#predicted_y = [0, 10, 20, 30, 40, 50]
#ys = [-3., -1., 1., 3., 5., 7.]
#loss = mean((predicted_y - ys)^2)  = mean(([0, 10, 20, 30, 40, 50] - [-3., -1., 1., 3., 5., 7.])^2)
# = mean(([3, 11, 19, 27, 35, 43])^2) = mean([9, 121, 361, 729, 1225, 1849]) = 715.6667
# so loss is correct
# w1 = w0 - learning_rate * d(loss)/dw [ see image ]
# b1 = b0 - learning_rate * d(loss)/db [ see image ]
# w1 = 10-0.09 * d(loss)/dw = -0.41
# b1 = 10-0.09 * d(loss)/db = 5.86

