import matplotlib.pyplot as plt
from validation import val
from model_train import *


# Define the NEW input data
x_new = torch.linspace(0, 2, new_data_points)[:, None]

y_pred_new = model(x_new.to(device))


plt.plot(X, F(X)) 
plt.plot(x_new.cpu().detach().numpy(), y_pred_new.cpu().detach().numpy(),'k', lw=2, label= "NN Prediction ")
plt.plot(x_new, F(x_new), 'ro', markersize = 4, label = 'Data Points')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize='x-small')
plt.show()