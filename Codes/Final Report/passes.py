# -----Setting Loss Function and Optimiser-----
criterion = torch.nn.MSELoss() # Loss Criterion (Mean Squared Error)
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -----Training Loop (A single Forward and Backward pass per epoch)-----
for epoch in range(num_epochs): # Sets a loop for each epoch
    y_pred = model(X_train) # Initiates forward pass
    loss = criterion(y_pred, Y_train) # Computes loss from the prediction
    optimiser.zero_grad() # Rests gradiets per epoch
    loss.backward() # Calculates gradients of loss with backpropagation
    optimiser.step() # Updates model parameters using gradients 