Z = torch.linspace(0, 1, 1000)[:,None]

with torch.no_grad():
    y_pred_new = model(Z.to(device)) # Define the NEW input data

# Define the NEW input data
plt.figure(figsize=(10,3))
plt.plot(Z, F(Z)) 
plt.plot(Z.cpu().detach().numpy(), y_pred_new.cpu().detach().numpy(),'k', lw=2, label= "NN Prediction ") # Prediction made by the network
plt.plot(X_train.cpu().detach().numpy(), Y_train.cpu().detach().numpy(), 'bo', markersize = 4, label = 'Data Points') # Data points used to train the network
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='x-small'); plt.show()