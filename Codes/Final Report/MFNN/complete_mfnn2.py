from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch


# Globally Shared Model(s) Hyperparameters
lf_num_epochs = 600
hf_num_epochs = 600
MF_epochs = 600

hidden_dims = ([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])

LF_input_dim = 1; LF_output_dim = 1
HF_input_dim = 1; HF_output_dim = 1
MF_input_dim = 2; MF_output_dim = 1

#-----Setting up the synthetic data set-----
# Setting the complxity of the synthetic functions 
def LF(y): # LF output function  
    return 0.5 * (6 * y - 2) ** 2 * np.sin(12 * y - 4) + 10 * (y - 0.5) - 5

def HF(y): # HF output function
    return (6 * y - 2)**2 * np.sin(12 * y - 4)

# Setting the Model's total training & testing data sets
LF_training_size = 20
HF_training_size = 6
LF_TS = torch.linspace(0, 1, LF_training_size)[:,None]
HF_TS = torch.linspace(0, 1, HF_training_size)[:,None] # forms a range of points between 0 and 1

# Splitting the data sets into training and testing sets per fidelity
X_LF_train, X_LF_test, Y_LF_train, Y_LF_test = train_test_split(LF_TS, LF(LF_TS), test_size=0.8, shuffle=156)
X_HF_train, X_HF_test, Y_HF_train, Y_HF_test = train_test_split(HF_TS, HF(HF_TS), test_size=0.8, shuffle=156)

# Forming the equally spaced training and testing sets per fidelity
X_LF_train = LF_TS; Y_LF_train = LF(LF_TS) 
X_HF_train = HF_TS; Y_HF_train = HF(HF_TS)

# Low Fidelity Network



class LowFidelityNetwork(torch.nn.Module):
  def __init__(self, hidden_dims, LF_input_dim, LF_output_dim):
    super().__init__()
    self.fc1 = nn.Linear(LF_input_dim, hidden_dims[0])
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
    self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
    self.fc5 = nn.Linear(hidden_dims[3], hidden_dims[4])
    self.fc6 = nn.Linear(hidden_dims[4], hidden_dims[5])
    self.fc7 = nn.Linear(hidden_dims[5], hidden_dims[6])
    self.fcEND = nn.Linear(hidden_dims[6], LF_output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    x = torch.relu(self.fc5(x))
    x = torch.relu(self.fc6(x))
    x = torch.relu(self.fc7(x))
    x = self.fcEND(x)
    return x

LF_model = LowFidelityNetwork(hidden_dims, LF_input_dim, LF_output_dim)

# Training
LF_model.train()

LF_batch_size = 100

# LF_loss criterion and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(LF_model.parameters(), lr=0.0005) # weight_decay=1e-5


for epoch in range(lf_num_epochs):    
    permutation = torch.randperm(X_LF_train.size()[0])

    for i in range(0,X_LF_train.size()[0], LF_batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+LF_batch_size]
        batch_x, batch_y = X_LF_train[indices], Y_LF_train[indices]
        outputs = LF_model.forward(batch_x)
        LF_loss = criterion(outputs,batch_y)
        LF_loss.backward()
        optimizer.step()

    if (epoch+1) % 600 == 0:
        print(f'Epoch [{epoch+1}/{hf_num_epochs}], Loss: {LF_loss.item():.4f}')

# High Fidelity Model

class HighFidelityNetwork(torch.nn.Module):
  def __init__(self, hidden_dims, HF_input_dim, HF_output_dim):
    super().__init__()
    self.fc1 = nn.Linear(HF_input_dim, hidden_dims[0])
    # self.bn1 = nn.BatchNorm1d(hidden_dims[0])
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
    self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
    self.fc5 = nn.Linear(hidden_dims[3], hidden_dims[4])
    self.fc6 = nn.Linear(hidden_dims[4], hidden_dims[5])
    self.fc7 = nn.Linear(hidden_dims[5], hidden_dims[6])
    self.fcEND = nn.Linear(hidden_dims[7], HF_output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    # x = self.bn1(x)
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    x = torch.relu(self.fc5(x))
    x = torch.relu(self.fc6(x))
    x = torch.relu(self.fc7(x))
    x = self.fcEND(x)
    return x

HF_model = HighFidelityNetwork(hidden_dims, HF_input_dim, HF_output_dim)

# %%


# Training
HF_model.train()
HF_batch_size = 45

# HF_loss criterion and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(HF_model.parameters(), lr=0.0001, weight_decay=1e-4) # weight_decay=1e-5

for epoch in range(hf_num_epochs):
    permutation = torch.randperm(X_HF_train.size()[0])
    
    for i in range(0,X_HF_train.size()[0], HF_batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+HF_batch_size]
        batch_x, batch_y = X_HF_train[indices], Y_HF_train[indices]
    
        outputs = HF_model.forward(batch_x)
        HF_loss = criterion(outputs,batch_y)
        HF_loss.backward()
        optimizer.step()


# Multi-Fidelity Model

# %%
L1mean = LF_model(X_HF_train)

L2train = torch.hstack((X_HF_train, L1mean)) # think of the house price example (sqr feet, rooms, garden, etc.)
print(L2train.shape)
# %%

class MultiFidelityNetwork(torch.nn.Module):
  def __init__(self, hidden_dims, MF_input_dim, MF_output_dim):
    super().__init__()
    self.fc1 = nn.Linear(MF_input_dim, hidden_dims[0])
    
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
    self.fc4 = nn.Linear(hidden_dims[2], hidden_dims[3])
    self.fc5 = nn.Linear(hidden_dims[3], hidden_dims[4])
    self.fc6 = nn.Linear(hidden_dims[4], hidden_dims[5])
    self.fc7 = nn.Linear(hidden_dims[5], hidden_dims[6])
    self.fcEND = nn.Linear(hidden_dims[6], MF_output_dim)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    x = torch.relu(self.fc5(x))
    x = torch.relu(self.fc6(x))
    x = torch.relu(self.fc7(x))
    x = self.fcEND(x)
    return x

MF_model = MultiFidelityNetwork(hidden_dims, MF_input_dim, MF_output_dim)

# %%
MF_model.train()

# MF_loss criterion and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(MF_model.parameters(), lr=0.00005) # weight_decay=1e-5

for epoch in range(MF_epochs):

    # Training
    MF_y_pred = MF_model(L2train)
    MF_loss = criterion(MF_y_pred, Y_HF_train)
    optimizer.zero_grad()
    MF_loss.backward(retain_graph=True)
    optimizer.step()

# -----Prediction stage-----
LF_model.eval(); HF_model.eval(); MF_model.eval() # Put the models into evaluation (prediction stage)
Z = torch.linspace(0, 1, 1000)[:,None] # New X_new_input data

with torch.no_grad(): 
    y_LF_pred = LF_model(Z)
    y_HF_pred = HF_model(Z) 
    
    # Step 4: Add X and output of the LF model (similar to Step 2)
    L2test = torch.hstack((Z, y_LF_pred))
    y_MF_pred = MF_model(L2test) 
    
