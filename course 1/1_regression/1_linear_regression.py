import math
import numpy as np
# Linear Regression Example

x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

# above x and Y are given. Find the function that best fits the data.
# We will use the equation of a line: y = wx + b
#start with the assumption that w=3 and b=-1
# we will calculate the root mean square error (rmse) to see how well our assumption fits the data
w = 3
b = -1
myY = []

for i in range(len(x)):
    myY.append(w * x[i] + b)

sse=0
sse += sum((y[i] - myY[i]) ** 2 for i in range(len(y)))
print(f"RMSE is :{np.round(math.sqrt(sse),3)}")


