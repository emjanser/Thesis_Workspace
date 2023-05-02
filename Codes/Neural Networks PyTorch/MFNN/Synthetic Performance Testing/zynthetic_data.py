import numpy as np
import torch

# Synthetic Data - Which gets more complex step by step 
# (a) Continuous functions with linear relationship
def QL_a(y):
    return 0.5 * (6 * y - 2) ** 2 * np.sin(12 * y - 4) + 10 * (y - 0.5) - 5

def QH_a(y):
    return (6 * y - 2)**2 * np.sin(12 * y - 4)
 
# (c) Continuous functions with nonlinear relationship
def QL_c(y):
    return 0.5 * (6 * y - 2)**2 * np.sin(12 * y - 4) + 10 * (y - 0.5) - 5

def QH_c(y):
    return (6 * y - 2)**2 * np.sin(12 * y - 4) - 10 * (y - 1)**2

# (e) Phase-Shifted Osicallations
def QL_e(y):
    return np.sin(8 * np.pi * y)

def QH_e(y):
    return y**2 + QL_e(y + np.pi/10)

# (f) Different Periodicities
def QL_f(y):
    return np.sin(6 * np.sqrt(2) * np.pi * y)

def QH_f(y):
    return np.sin(8 * np.pi * y + np.pi / 10)

# (b) Discontinuous functions with linear relationship
def LF(y):
    zeros = torch.zeros_like(y)
    half = torch.ones_like(y) * 0.5
    y_0 = torch.where(y <= half, 0.5 * (6 * y - 2)**2 * torch.sin(12 * y - 4) + 10 * (y - 0.5), zeros)
    y_1 = torch.where(half < y, 3 + 0.5 * (6 * y - 2)**2 * torch.sin(12 * y - 4) + 10 * (y - 0.5), y_0)
    return y_1

def HF(y):
    zeros = torch.zeros_like(y)
    half = torch.ones_like(y) * 0.5
    y_0 = torch.where(y <= half, 2*LF(y) - 20*y + 20, zeros)
    y_1 = torch.where(half < y, 4 + 2*LF(y) - 20*y + 20, y_0)
    return y_1

# (d) Continuous oscillation functions with nonlinear relationship
def QL_c(y):
    if 0 <= y <= 1:
        return np.sin(8 * np.pi * y)
    else:
        return ValueError("(d) Range Error")

def QH_c(y):
    if 0 <= y <= 1:
        return (y - np.sqrt(2)) * QL_c(y) ** 2
    else:
        return ValueError("(d) Range Error")