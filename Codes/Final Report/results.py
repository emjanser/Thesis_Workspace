new_X_input = torch.linspace(0, 1, 1000)[:,None] # Random ranged data array for new X_input data
with torch.no_grad():
    y_pred_new = model(new_X_input) # Define the NEW input data
# Plotting the results and change in loss using matplotlib (Provided in Appendix B)