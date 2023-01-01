import torch
import math

class HighFidelityModel(torch.nn.Module):
    def init(self, input_dim, hidden_dim, output_dim):
        super().init()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class LowFidelityModel(torch.nn.Module):
    def init(self, input_dim, hidden_dim, output_dim):
        super().init()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# Define the multifidelity model
class MultiFidelityModel(torch.nn.Module):
    def init(self, input_dim, hidden_dim, output_dim):
        super().init()

        self.high_fidelity_model = HighFidelityModel(input_dim, hidden_dim, output_dim)
        self.low_fidelity_model = LowFidelityModel(input_dim, hidden_dim, output_dim)
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    
def forward(self, x, fidelity):
    if fidelity == "high":
        return self.high_fidelity_model(x)

    elif fidelity == "low":
        return self.low_fidelity_model(x)
    else:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the function to predict using the multifidelity model
def multi_fidelity_predict(x, fidelity, threshold):
    y_pred = model(x, fidelity)

    # Calculate the error
    error = y_pred - y(x)
    expected_error = abs(error.mean())

# If the fidelity model is not accurate enough, use the multifidelity model to refine the estimate
    if abs(error) > threshold:
        y_pred = model(x, "mf")

    return y_pred

# Define the high and low fidelity data
X = torch.linspace(0, 1, 100)
y = lambda x: 1.8*torch.sin(x * (8 * math.pi))*2*x
y_lf = lambda x: 1.2*torch.sin(x * (6 * math.pi))*2*x

# Split the data into training and testing sets
x_train, x_test = torch.split(X, [70, 30])
y_train = y