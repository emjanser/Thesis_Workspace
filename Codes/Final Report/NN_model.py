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
