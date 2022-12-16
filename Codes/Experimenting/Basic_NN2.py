import numpy as np
import torch
import matplotlib.pyplot as plt

class NeuralNetwork(torch.nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim):
    super().__init__()
    self.fc1 = torch.nn.Linear(input_dim, hidden_dims[0])
    self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
    ...
    self.fc99 = torch.nn.Linear(hidden_dims[98], hidden_dims[99])
    self.fc100 = torch.nn.Linear(hidden_dims[99], output_dim)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    ...
    x = self.fc99(x)
    x = self.fc100(x)
    return x

# Define the input and output dimensions
input_dim = 1
output_dim = 1

# Define the number of hidden units in the network
#hidden_dims = [32, 64, 128, ... ,256, 512,1014]
# hidden_dims = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
#31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,57,58,60,60,60,60,60,
# 60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,60,1000,2]

# Create the neural network
model = NeuralNetwork(input_dim, hidden_dims, output_dim)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Define the input data
x = torch.randn(1000, input_dim)

y = x.pow(2).sum(1)
y = y.view(-1, 1)

# Define the number of epochs
num_epochs = 60

# Keep track of the losses
losses = []

# Train the model
for epoch in range(num_epochs):
  # Forward pass
  y_pred = model(x)

  # Compute the loss
  loss = criterion(y_pred, y)
  losses.append(loss.item())

  # Zero the gradients
  optimizer.zero_grad()

  # Compute the gradients
  loss.backward()

  # Update the parameters
  optimizer.step()

# Plot the losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Define the input data---------------------------------------
x_new = torch.randn(1000, input_dim)



# Get the predictions with the new data
y_pred_new = model(x_new)
y_pred_new = y_pred_new[:x.shape[0]]

print(x.shape)
print(y_pred_new.shape)

# Evaluate the network on the validation set --------------------
val_outputs = model(x)
val_loss = criterion(val_outputs, y)

# Print the validation loss
print(val_loss)

# Use the network to make predictions
predictions = model(x)

# Plot the predictions against the true values
plt.plot(x.numpy(), y.numpy(), 'o', label='True data')
plt.plot(x.numpy(), predictions.detach().numpy(), 'o', label='Validation Predictions')
plt.plot(x.numpy(), y_pred_new.detach().numpy(), 'o', label='New Data Predictions')

plt.legend()
plt.show()


"""
import torch
import matplotlib.pyplot as plt

class MultiFidelityNN(torch.nn.Module):
  def __init__(self, input_dim, output_dim):
    super().__init__()
    # Define 100 hidden layers
    self.fc1 = torch.nn.Linear(input_dim, hidden_dims[0])
    self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
    ...
    self.fc99 = torch.nn.Linear(hidden_dims[98], hidden_dims[99])
    self.fc100 = torch.nn.Linear(hidden_dims[99], output_dim)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    ...
    x = self.fc99(x)
    x = self.fc100(x)
    return x

# Define the input and output dimensions
input_dim = 2
output_dim = 1

# Define the number of hidden units in the network
hidden_dims = [32, 64, 128, 256]

# Create the neural network
model = MultiFidelityNN(input_dim, output_dim)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Define the input data
x = torch.randn(1000, input_dim)
y = x.pow(2).sum(1)

# Define the number of epochs
num_epochs = 1000

# Keep track of the losses
losses = []

# Define the input data
x_new = torch.randn(10, input_dim)

# Get the predictions
y_pred_new = model(x_new)

# Train the model
for epoch in range(num_epochs):
  # Forward pass
  y_pred = model(x)

  # Compute the loss
  loss = criterion(y_pred, y)
  losses.append(loss.item())

  # Zero the gradients
  optimizer.zero_grad()

  # Compute the gradients
  loss.backward()

  # Update the parameters
  optimizer.step()

# Plot the losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Get the predictions on the new data
y_pred_new = model(x_new)

# Reshape y_pred_new to have the same number of rows as x
y_pred_new = y_pred_new[:x.shape[0]]

# Check the shapes of x and y_pred_new
print(x.shape)  # prints (1000, 1)
print(y_pred_new.shape)  # prints (1000, 1)

# Plot the predictions
plt.plot(y_pred_new.detach().numpy(), 'o')
plt.show



"""