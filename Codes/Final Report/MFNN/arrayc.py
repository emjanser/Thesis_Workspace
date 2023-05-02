LF_pred = LF_model(X_HF_train) # LFNN prediction of LF data
mixed_fidelity_array = torch.hstack((X_HF_train, L1mean)) # stacking arrays side to side

# MFNN NN Model Structure (Identical to a typical NN - Appendix B)
MF_model = MultiFidelityNetwork(hidden_dims, MF_input_dim, MF_output_dim)

# ------MFNN Training Loop-----
MF_model.train()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(MF_model.parameters(), lr=0.00005)

for epoch in range(MF_epochs):
    MF_y_pred = MF_model(mixed_fidelity_array)
    MF_loss = criterion(MF_y_pred, Y_HF_train) # Trains the MFNN model using the mixed array and HF training set
    optimizer.zero_grad()
    MF_loss.backward(retain_graph=True)
    optimizer.step()