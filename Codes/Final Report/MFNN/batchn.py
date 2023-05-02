class HighFidelityNetwork(torch.nn.Module):
  def __init__(self, hidden_dims, HF_input_dim, HF_output_dim):
    super().__init__()
    self.fc1 = nn.Linear(HF_input_dim, hidden_dims[0])
    self.bn1 = nn.BatchNorm1d(hidden_dims[0]) # batch normalisation
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
    self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
    self.fcEND = nn.Linear(hidden_dims[3], HF_output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    skip_connection = x  # skip connection
    x = self.bn1(x)
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    x = x + skip_connection
    x = self.fcEND(x)
    return x