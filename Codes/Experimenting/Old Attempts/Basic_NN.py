import torch
import matplotlib.pyplot as plt

class NeuralNetwork(torch.nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim):
    super().__init__()
    self.fc1 = torch.nn.Linear(input_dim, hidden_dims[0])
    self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
    self.fc3 = torch.nn.Linear(hidden_dims[1], hidden_dims[2])
    self.fc4 = torch.nn.Linear(hidden_dims[2], hidden_dims[3])
    self.fc5 = torch.nn.Linear(hidden_dims[3], hidden_dims[4])
    self.fc6 = torch.nn.Linear(hidden_dims[4], hidden_dims[5])
    self.fc7 = torch.nn.Linear(hidden_dims[5], hidden_dims[6])
    self.fc8 = torch.nn.Linear(hidden_dims[6], output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    x = torch.relu(self.fc5(x))
    x = torch.relu(self.fc6(x))
    x = torch.relu(self.fc7(x))
    x = self.fc8(x)
    return x

# Define the input and output dimensions
input_dim = 1
output_dim = 1

# Define the number of hidden units in the network
hidden_dims = [32, 64, 128, 256, 512, 1014, 2028]

# Create the neural network
model = NeuralNetwork(input_dim, hidden_dims, output_dim)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss( )
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
y_pred_new = model(x_new)

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
class MultiFidelityNN(torch.nn.Module):
  def __init__(self, input_dim, hidden_dims, output_dim):
    super().__init__()
    self.fc1 = torch.nn.Linear(input_dim, hidden_dims[0])
    self.fc2 = torch.nn.Linear(hidden_dims[0], hidden_dims[1])
    self.fc3 = torch.nn.Linear(hidden_dims[1], hidden_dims[2])
    ...
    self.fc_hundreds = torch.nn.Linear(hidden_dims[-2], hidden_dims[-1])
    self.fc_output = torch.nn.Linear(hidden_dims[-1], output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    ...
    x = torch.relu(self.fc_hundreds(x))
    x = self.fc_output(x)
    return x


"""