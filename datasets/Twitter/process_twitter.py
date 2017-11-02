#!/usr/bin/python3

import sys, os
import math
import numpy as np
#import matplotlib.pyplot as plt

X = []
Y = []

def transform_y(y):
  return math.log(y + 1.0)

with open("Twitter.data") as f:
  for line in f.readlines():
    nums = list(map(float, line.strip().split(",")))
    X.append(nums[:-1])
    Y.append(transform_y(nums[-1]))

x_norms = [sum(list(map(abs, x))) for x in X]
max_x_norm = max(x_norms)
y_norms = list(map(abs,Y))
max_y_norm = max(y_norms)

print("n = " + str(len(X)))
print("d = " + str(len(X[0])))
print("")
print("X norms:")
print("Max 1-norm: " + str(max_x_norm))
print("Average 1-norm: " + str(sum(x_norms)/len(x_norms)))
x_norms.sort()
print("Median 1-norm: " + str(x_norms[int(len(x_norms)/2)]))
print("")
print("Y norms:")
print("Max 1-norm: " + str(max_y_norm))
print("Average 1-norm: " + str(sum(y_norms)/len(y_norms)))
y_norms.sort()
print("Median 1-norm: " + str(y_norms[int(len(y_norms)/2)]))

#plt.figure()
#plt.plot(y_norms,np.linspace(0,1,len(y_norms)))
#plt.title("Y 1-norm CDF")
#plt.show()

with open("twitter_dataset.txt", "w") as f:
  for x,y in zip(X,Y):
    xhat = [x/max_x_norm for x in x]
    yhat = y/max_y_norm
    f.write(" ".join(list(map(str, xhat))))
    f.write(" ")
    f.write(str(yhat))
    f.write("\n")

