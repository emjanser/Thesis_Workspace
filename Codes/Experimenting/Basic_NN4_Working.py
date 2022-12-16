import torch
import math
import numpy as np
import matplotlib.pyplot as plt

#* Neural Network Main Parameters
num_epochs = 200
input_dim = 1
output_dim = 1
hidden_dims = 2*([32, 64, 128, 256, 512, 1014, 2028]) # Defining the number of nodes in the hidden layers

#Model Training & Testing Points Settings
model_training_points = 400
model_testing_points = 150

# New Data Point Parameters
new_data_points = 30

#* Forming the Training and Testing Points
def y(x): 
    return 1.8*torch.sin(x * (8 * math.pi))*2*x

# Training data
X = torch.linspace(0, 1, model_training_points)
x_train = np.random.permutation(X)
x_train = torch.from_numpy(x_train)
x_train = x_train[:, None]

y_train = y(x_train)
y_train = y_train.view(-1, 1)



# Testing data 
X = torch.linspace(0, 1, model_testing_points)
x_test = np.random.permutation(X)
x_test = torch.from_numpy(x_test) 
x_test = x_test[:, None]

y_test = y(x_test)
y_test = y_test.view(-1, 1)

#* Setting the NN model
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

model = NeuralNetwork(input_dim, hidden_dims, output_dim)

# Defining the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Keep track of the losses to plot it in the future
losses = []

# Train the model
for epoch in range(num_epochs):
  # Forward pass
  y_pred = model(x_train)

  # Computing the loss
  loss = criterion(y_pred, y_train)
  losses.append(loss.item())

  optimizer.zero_grad()

  loss.backward()

  optimizer.step()

# Plotting the losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Evaluating the network on the validation set 
val_outputs = model(x_test)
val_loss = criterion(val_outputs, y_test)
print(val_loss)

# Using the network to make predictions-------------------- Model Training is Complete--------------------------------------
predictions = model(x_test)


# ----------------------------------------------------------Define the NEW input data---------------------------------------
X = torch.linspace(0, 1, new_data_points)
x_new = np.random.permutation(X)
x_new = torch.from_numpy(x_new)
x_new = x_new[:, None]

y_pred_new = model(x_new)


# Plotting the predictions against the true values
plt.plot(x_train.numpy(), y_train.numpy(), 'o', label='True data')
plt.plot(x_test.numpy(), predictions.detach().numpy(), 'o', label = 'Validation Predictions')
plt.plot(x_new.numpy(), y_pred_new.detach().numpy(), 'o', label = 'New Data Predictions')

plt.legend()
plt.show()