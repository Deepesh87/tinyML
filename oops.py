import numpy as np 
import pandas as pd
import math

class testing():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def add(self):
        return self.a + self.b

    def multiple(self):
        return self.a * self.b


a = 10
b = 20
test = testing(a, b)
print("Addition:", test.add())
print("Multiplication:", test.multiple())
