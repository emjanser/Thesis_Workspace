from data import *

class Network(torch.nn.Module):
  def __init__(self, input_dim, HF_hidden_dims, output_dim):
    super().__init__()
    self.fc1 = torch.nn.Linear(input_dim, HF_hidden_dims[0])
    self.fc2 = torch.nn.Linear(HF_hidden_dims[0], HF_hidden_dims[1])
    self.fc3 = torch.nn.Linear(HF_hidden_dims[1], HF_hidden_dims[2])
    self.fc4 = torch.nn.Linear(HF_hidden_dims[2], HF_hidden_dims[3])
    self.fc5 = torch.nn.Linear(HF_hidden_dims[3], output_dim)
    # self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    # x = self.dropout(x)
    x = self.fc5(x)
    return x

model = Network(input_dim, hidden_dims, output_dim).to(device)



# Training
losses = [] 

# Loss criterion and optimizer
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters())


for epoch in range(num_epochs):
  y_pred = model(X_train.to(device))
  loss = criterion(y_pred, Y_train.to(device))
  losses.append(loss.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

for param in model.parameters():
  param.requires_grad = True