# 'Pooling', which compresses your image, further emphasising the features.
# 'Convolution', which applies a filter to your image, emphasising certain features.
# 'Activation', which applies a non-linear function to your image, allowing the model to learn complex patterns.
# 'Flattening', which converts your image into a 1D array, so it can be processed by the model.
# 'Dense', which applies a fully connected layer to your image, allowing the model to learn complex patterns.

import cv2
import numpy as np
import tensorflow as tf


#============================LETS DO USING A NORMAL DNN========================================
# This code uses a normal DNN to classify images from the Fashion MNIST dataset.
# Load an image from the Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (val_images, val_labels) = mnist.load_data()
training_images=training_images / 255.0
val_images=val_images / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),  # Flatten the 28x28 images to a 1D array of 784 pixels
  tf.keras.layers.Dense(20, activation=tf.nn.relu),   # Hidden layer with 20 neurons and ReLU activation
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)   # Output layer with 10 neurons for each class (0-9)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, validation_data=(val_images, val_labels), epochs=20)
#accuracy is 89% on training and 87% on validation

#=============================LETS DO USING A CONVOLUTIONAL NEURAL NETWORK========================================
# This code uses a convolutional neural network to classify images from the Fashion MNIST dataset.
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (val_images, val_labels) = mnist.load_data()

#3D tensor (samples, height, width)

#60k is The number of training images in the dataset. 28: The height of each image in pixels.28: 
# The width of each image in pixels. 1: The number of color channels. 1 means grayscale (black and white).
training_images = training_images.reshape(60000, 28, 28, 1) # reshape to add a channel dimension for grayscale images 
training_images = training_images / 255.0  # Normalize the images to a range of  0 to 1
val_images = val_images.reshape(10000, 28, 28, 1) # reshape to add a channel dimension for grayscale images
val_images = val_images / 255.0  # Normalize the images to a range of 0 to 1


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)), #Conv2D: Applies 64 filters of size (3×3) to the input image.
  tf.keras.layers.MaxPooling2D(2, 2), #Downsamples the feature maps by taking the maximum value over a 2x2 window.
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'), #Another convolutional layer, same as before, further extracting features.
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),  #Converts the 3D tensor from the convolutional layers into a 1D vector for the Dense (fully connected) layers.
  tf.keras.layers.Dense(20, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax') #Softmax: Turns raw scores into probabilities across 10 classes.
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, validation_data=(val_images, val_labels), epochs=100) 
# accuracy has gone up to 96% on training and 90% on validation with 20 epochs
# with epoch of 100 it overfits to 99% on training and 91% on validation
