
# -----Importing Essential Packages-----
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CUDA device configuration

# -----Model Hyperparameters Initialisation-----
num_epochs = 1000
n_per_layer = ([50, 100, 150]) 
input_dim = 1; output_dim = 1 
learning_rate = 0.0001 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# -----Model Structure Initialisation with 2 Hidden Layers-----
class Network(torch.nn.Module): 
  def __init__(self, n_per_layer, input_dim, output_dim): # Layer setup
    super().__init__()
    self.fc1 = nn.Linear(input_dim, n_per_layer[0])  # Input Layer
    self.fc2 = nn.Linear(n_per_layer[0], n_per_layer[1]) # Hidden Layer 1
    self.fc3 = nn.Linear(n_per_layer[1], n_per_layer[2]) # Hidden Layer 2
    self.fc4 = nn.Linear(n_per_layer[2], output_dim) # Output Layer

  def forward(self, x): # Activation function after each layer (ReLu)
    x = torch.relu(self.fc1(x)) 
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = self.fc4(x)
    return x

model = Network(n_per_layer, input_dim, output_dim) # Calling Python class as to use the model when needed
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# -----Setting Loss Function and Optimiser-----
criterion = torch.nn.MSELoss() # Loss Criterion (Mean Squared Error)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -----Training Loop (A single Forward and Backward pass per epoch)-----
for epoch in range(num_epochs):
    y_pred = model(X_train) 
    loss = criterion(y_pred, Y_train) 
    optimiser.zero_grad() 
    loss.backward() 
    optimiser.step()