def F(y):
    return np.sin(8 * np.pi * y) # Target function (synthetic data)

X_train = torch.linspace(0, 1, training_data)[:,None] # Model's training and testing data sets
Y_train = F(X_train)
