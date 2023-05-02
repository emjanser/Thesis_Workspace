with torch.no_grad(): # prevents the gradients of the models from changing during prediction
    y_LF_pred = LF_model(X_new_input) # predtion obtained from the singular NN model
    
    prediction_array = torch.hstack((X_new_input, y_LF_pred)) # combining new X_input with LF prediction
    y_MF_pred = MF_model(prediction_array) # Running the combined array through MFNN