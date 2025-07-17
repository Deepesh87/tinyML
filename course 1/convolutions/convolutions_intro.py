# 'Pooling', which compresses your image, further emphasising the features.
# 'Convolution', which applies a filter to your image, emphasising certain features.
# 'Activation', which applies a non-linear function to your image, allowing the model to learn complex patterns.
# 'Flattening', which converts your image into a 1D array, so it can be processed by the model.
# 'Dense', which applies a fully connected layer to your image, allowing the model to learn complex patterns.

import cv2
import numpy as np
from scipy import datasets
i = datasets.ascent().astype(np.int32)

#see the image
import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
print(size_x)
size_y = i_transformed.shape[1]
print(size_y)
# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight
# lines.

#Experiment with different values for fun effects.
filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]

# A couple more filters to try for fun!
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# If all the digits in the filter don't add up to 0 or 1, you 
# should probably do a weight to get it to do so
# so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
weight  = 1

# let's create a convolution. We will iterate over the image, leaving a 1 pixel margin, and multiply out each of the neighbors of the current pixel 
# by the value defined in the filter.
# i.e. the current pixel's neighbor above it and to the left will be multiplied by the top left item in the filter etc. etc.
#  We'll then multiply the result by the weight, and then ensure the result is in the range 0-255
# Finally we'll load the new value into the transformed image.

for x in range(1,size_x-1):
  for y in range(1,size_y-1):
      convolution = 0.0
      convolution = convolution + (i[x - 1, y-1] * filter[0][0])
      convolution = convolution + (i[x, y-1] * filter[1][0])
      convolution = convolution + (i[x + 1, y-1] * filter[2][0])
      convolution = convolution + (i[x-1, y] * filter[0][1])
      convolution = convolution + (i[x, y] * filter[1][1])
      convolution = convolution + (i[x+1, y] * filter[2][1])
      convolution = convolution + (i[x-1, y+1] * filter[0][2])
      convolution = convolution + (i[x, y+1] * filter[1][2])
      convolution = convolution + (i[x+1, y+1] * filter[2][2])
      convolution = convolution * weight
      if(convolution<0):
        convolution=0
      if(convolution>255):
        convolution=255
      i_transformed[x, y] = convolution

# see the transformed image
# Plot the image. Note the size of the axes -- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()   

# Pooling
# As well as using convolutions, pooling helps us greatly in detecting features. The goal is to reduce the overall amount of information 
# in an image, while maintaining the features that are detected as present.
# There are a number of different types of pooling, but for this lab we'll use one called MAX pooling.
# The idea here is to iterate over the image, and look at the pixel and it's immediate neighbors to the '
# 'right, beneath, and right-beneath. Take the largest (hence the name MAX pooling) of them and load it'
# ' into the new image. Thus the new image will be 1/4 the size of the old -- with the dimensions on X '
# 'and Y being halved by this process. You'll see that the features get maintained despite this compression!
new_x = int(size_x/4)
new_y = int(size_y/4)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 4):
  for y in range(0, size_y, 4):
    pixels = []
    #row 0
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x+1, y])
    pixels.append(i_transformed[x+2, y])
    pixels.append(i_transformed[x+3, y])
    #row 1
    pixels.append(i_transformed[x, y+1])
    pixels.append(i_transformed[x+1, y+1])
    pixels.append(i_transformed[x+2, y+1])
    pixels.append(i_transformed[x+3, y+1])
    #row 2
    pixels.append(i_transformed[x, y+2])
    pixels.append(i_transformed[x+1, y+2])
    pixels.append(i_transformed[x+2, y+2])
    pixels.append(i_transformed[x+3, y+2])
    #row 3
    pixels.append(i_transformed[x, y+3])
    pixels.append(i_transformed[x+1, y+3])
    pixels.append(i_transformed[x+2, y+3])
    pixels.append(i_transformed[x+3, y+3])
    pixels.sort(reverse=True) #sort in descending order
    #take the first pixel, which is the largest
    newImage[int(x/4),int(y/4)] = pixels[0]

# Plot the image. Note the size of the axes -- now 128 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()   
