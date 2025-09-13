# Use the helper./py file to understand the functions
import tensorflow as tf
import matplotlib.pyplot as plt
# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

xs = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0, 1.0, 3.0, 5.0, 7.0]
# above x and y are given. Find the function that best fits the data.
# We will use the equation of a line: y = wx + b
#start with the assumption of a W and B
# we will calculate the root mean square error (RMSE) to see how well our assumption fits the data

# Define our initial guess
INITIAL_W = 10.0
INITIAL_B = 10.0
LEARNING_RATE=0.09

# Define our loss function
def loss(predicted_y, target_y):
  """
  implements the Mean Squared Error (MSE) loss function
  """
  return tf.reduce_mean(tf.square(predicted_y - target_y))

# Define our simple linear regression model
class Model(object):
  def __init__(self):
    # Initialize the weights
    self.w = tf.Variable(INITIAL_W)
    self.b = tf.Variable(INITIAL_B)

  def __call__(self, x):
    return self.w * x + self.b 
  

# Define our training procedure
def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)
    # Here is where you differentiate the model values with respect to the loss function
    dw, db = t.gradient(current_loss, [model.w, model.b])
    # And here is where you update the model values based on the learning rate chosen
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)
    return current_loss

  
# Instantiate our model
model = Model()

# Collect the history of w-values and b-values to plot later
list_w, list_b = [], []
epochs = range(50)
losses = []
for epoch in epochs:
  list_w.append(model.w.numpy())
  list_b.append(model.b.numpy())
  current_loss = train(model, xs, ys, learning_rate=LEARNING_RATE)
  losses.append(current_loss)
  print(f"Epoch {epoch:2d}: w={list_w[-1]:1.2f} b={list_b[-1]:1.2f}, " 
        f"loss={current_loss:2.5f}")
  

# Plot the w-values and b-values for each training Epoch against the true values
TRUE_w = 2.0
TRUE_b = -1.0
#save the plot

plt.plot(epochs, list_w, 'r', epochs, list_b, 'b')
plt.plot([TRUE_w] * len(epochs), 'r--', [TRUE_b] * len(epochs), 'b--')
plt.legend(['w', 'b', 'True w', 'True b'])
plt.savefig('linear_regression_plot.png')


# Read more here
# https://www.tensorflow.org/api_docs/python/tf/GradientTape
