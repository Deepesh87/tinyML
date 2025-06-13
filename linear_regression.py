import math
# Linear Regression Example
w = 3
b = -1

x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]
myY = []

for i in range(len(x)):
    myY.append(w * x[i] + b)

sse=0
sse+= sum((y[i] - myY[i]) ** 2 for i in range(len(y)))
print(f"rmse is :{math.sqrt(sse)}")


