from sklearn.linear_model import LinearRegression
import numpy as np

def train(x, y):
    model = LinearRegression().fit(x,y)
    
    return model

model = train(x,y)
x_new = 23.0
y_new = model.predict(x_new)
print(y_new)

#Bruh ofc the example is not going to work, you didn't give it a X and Y training set to train the thing... bruh