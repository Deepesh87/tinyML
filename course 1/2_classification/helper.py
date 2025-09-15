import tensorflow as tf
# 1 ==== datasets in TF, MNIST datset========================
data = tf.keras.datasets.mnist # has 60k training images and 10k validation images of handwritten digits (0-9)
(training_images, training_labels), (val_images, val_labels) = data.load_data()
print(training_images.shape) # (60000, 28, 28) 60k images of 28x28 pixels
print(training_labels.shape) # (60000,) 60k labels (0-9)
print(val_images.shape) # (10000, 28, 28) 10k images of 28x28 pixels
print(val_labels.shape) # (10000,) 10k labels (0-9)

print(training_images.min(), training_images.max()) # (0, 255) pixel values range from 0 to 255

print(training_images[0]) # print the first image as a 2D array of pixel values
print(training_labels[0]) # print the label of the first image (5)

# 2 ====  more on Activation Function  ========================
# reLU means Rectified Linear Unit, a common activation function in neural networks.
# It introduces non-linearity and helps the network learn complex patterns
# f(x)=max(0,x) , Efficient → Very simple math, fast to compute.

# softmax converts raw scores (logits) into probabilities that sum up to 1
# Softmax is an activation function used in the output layer of classification models when we have multiple classes.
# 
# What is OPTIMISER? -- tells model how to learn, i.e., how to update weights based on the loss function's output during training.
# Optimizer = algorithm that updates weights during training (by backpropagation)
# Adam (Adaptive Moment Estimation)
# Combines benefits of Momentum (remembers past gradients) and RMSProp (adapts learning rate for each weight).

# loss='sparse_categorical_crossentropy'
# Categorical Crossentropy = common for multi-class classification.
# It compares the predicted probability distribution (from Softmax) with the true class.
# Difference between categorical_crossentropy vs sparse_categorical_crossentropy:

# categorical_crossentropy → expects labels as one-hot encoded vectors.
# Example: digit 2 → [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# sparse_categorical_crossentropy → expects labels as integers.
# Example: digit 2 → 2
# That’s why sparse_categorical_crossentropy is easier with datasets like MNIST, where labels are just numbers. 

# Metrics are for monitoring performance (not directly used in training).


# 3 ====  Model loss function and metrics ====================



# 4 ====                   ========================
