from sklearn.model_selection import train_test_split
import numpy as np
import torch

# Synthetic Data points
total_points_dataset = 1000
new_data_points = 20

# Model Parameters
num_epochs = 1500
input_dim = 1
output_dim = 1
hidden_dims = ([900, 900, 900, 900, 1200, 1600, 2400])


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Model running via: {device}")

# Synthetic data function
def F(y):
    return 0.5 * (6 * y - 2)**2 * np.sin(12 * y - 4) + 10 * (y - 0.5) - 5


# Model's total dataset
X = torch.linspace(0, 2, total_points_dataset, dtype=torch.float32)[:,None]


# Splitting dataset
X_train, X_temp, Y_train, Y_temp = train_test_split(X, F(X), test_size=0.3, random_state=0)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=0)


# Transfering data to GPU for CUDA
X_train = X_train.to(device); X_val = X_val.to(device); X_test = X_test.to(device)
Y_train = Y_train.to(device); Y_val = Y_val.to(device); Y_test = Y_test.to(device)



# Dont forget about torch data needing requires_grad = True




""" Verification Step - Can be commented out. 
import matplotlib.pyplot as plt
plt.scatter(X_val, Y_val) # main result line we want to match with
plt.show()
print(type(X_val))
"""


"""
I can use this piece of code to work on the rest of the code after taking care of training / testing 

data = np.linspace(0, 1, model_training_points)[:,None]
split = int(model_training_points * 0.8) # 80% for training, 20% for testing

train_data = data[:split]
test_data = data[split:]

plt.plot(test_data, F(test_data))
plt.show()
"""