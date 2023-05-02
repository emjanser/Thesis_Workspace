# -----Importing Essential Packages-----
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # CUDA device configuration

# -----Model Hyperparameters Initialisation-----
num_epochs = 1000 # No. of epochs
input_dim = 1; output_dim = 1 # Input & Output dimensions
n_per_layer = ([500, 500, 500]) # Neurons per layer
learning_rate = 0.0001 # Learning rate of optimiser

# ------Setting up the synthetic training data set-----
def F(y):
    return np.sin(8 * np.pi * y) # Target function (synthetic data)

training_data = 13 # Number of training points
X_train = torch.linspace(0, 1, training_data)[:,None] # Model's training and testing data sets
Y_train = F(X_train)

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

# -----Setting Loss Function and Optimiser-----
criterion = torch.nn.MSELoss() # Loss Criterion (Mean Squared Error)
optimiser = torch.optim.Adam(model.parameters(), lr=0.005)

# -----Training Loop (A single Forward and Backward pass per epoch)-----
for epoch in range(num_epochs): # Sets a loop for each epoch
    y_pred = model(X_train) # Initiates forward pass
    loss = criterion(y_pred, Y_train) # Computes loss from the prediction
    optimiser.zero_grad() # Rests gradiets per epoch
    loss.backward() # Calculates gradients of loss with backpropagation
    optimiser.step() # Updates model parameters using gradients 
    
# -----Obtaining Predictions using the model-----
new_X_input = torch.linspace(0, 1, 1000)[:,None] # Random ranged data array for new X_input data
with torch.no_grad():
    y_pred_new = model(new_X_input) # Define the NEW input data
    
# Plotting the results and change in loss using matplotlib (Provided in Appendix B)
