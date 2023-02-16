import torch
import numpy as np
import math


""" Setting main parameters """
# Define the input and output dimensions
input_dim = 1
output_dim = 1

# Training data
X = torch.linspace(0, 1, 100)
x_train = np.random.permutation(X)
x_train = torch.from_numpy(x_train)

# Testing data 
X = torch.linspace(0, 1, 10)
x_test = np.random.permutation(X)
x_test = torch.from_numpy(x_test) 

# print(x_train)

def y(x): 
    return 1.8*torch.sin(x * (8 * math.pi))*2*x

# Training Data
y_train = (x_train)
x_train = (x_train, input_dim)


x_test = torch.randn(5, input_dim)
y_test = x_test.pow(2).sum(1)
print(y_test)
y_test = y_test.view(-1, 1)

print(y_test)

hidden_dims = 6*([32, 64, 128, 256, 512, 1014, 2028, 2028, 2028, 2028])

print(hidden_dims[:6])