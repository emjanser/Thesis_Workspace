import matplotlib.pyplot as plt
from model_train import *


# Evaluating the network on the validation set 
val_outputs = model(X_val)
val_loss = criterion(val_outputs, Y_val)
val = print(f" Valdation Loss of Model: {val_loss}")


# Valdation Loss Plot
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()