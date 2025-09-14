import tensorflow as tf

data = tf.keras.datasets.mnist # has 60k training images and 10k validation images of handwritten digits (0-9)
(training_images, training_labels), (val_images, val_labels) = data.load_data()
print(training_images.shape) # (60000, 28, 28) 60k images of 28x28 pixels
print(training_labels.shape) # (60000,) 60k labels (0-9)
print(val_images.shape) # (10000, 28, 28) 10k images of 28x28 pixels
print(val_labels.shape) # (10000,) 10k labels (0-9)

print(training_images.min(), training_images.max()) # (0, 255) pixel values range from 0 to 255

print(training_images[0]) # print the first image as a 2D array of pixel values
print(training_labels[0]) # print the label of the first image (5)


