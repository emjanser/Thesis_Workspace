import torch
import numpy


# Define the multi-fidelity neural network
class MultiFidelityNeuralNetwork(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super().__init__()

    # Define the low-fidelity neural network
    self.low_fidelity_network = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, output_dim)
    )

    # Define the high-fidelity neural network
    self.high_fidelity_network = torch.nn.Sequential(
      torch.nn.Linear(input_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, hidden_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(hidden_dim, output_dim)
    )

  def forward(self, inputs):
    # Use the low-fidelity network for inputs with low fidelity
    low_fidelity_inputs = inputs[inputs["fidelity"] == "low"]
    low_fidelity_outputs = self.low_fidelity_network(low_fidelity_inputs)

    # Use the high-fidelity network for inputs with high fidelity
    high_fidelity_inputs = inputs[inputs["fidelity"] == "high"]
    high_fidelity_outputs = self.high_fidelity_network(high_fidelity_inputs)

    # Concatenate the outputs from the low-fidelity and high-fidelity networks
    return torch.cat([low_fidelity_outputs, high_fidelity_outputs])


# Create an instance of the multi-fidelity neural network
model = MultiFidelityNeuralNetwork(input_dim=2, hidden_dim=10, output_dim=1)

# Create some inputs with low and high fidelity

# Create some inputs with low and high fidelity
inputs = torch.tensor([
  [[0, 0], [0]],  # Low-fidelity input
  [[1, 1], [1]], # High-fidelity input
  [[0, 1], [0]],  # Low-fidelity input
  [[1, 0], [1]], # High-fidelity input
], dtype=torch.float)



import matplotlib.pyplot as plt

# Use the model to make predictions
outputs = model(inputs)

# Plot the outputs
plt.plot(outputs.numpy())
plt.show()



