import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv(r"C:\Users\adity\Downloads\CarPrices\CarPrice_Assignment.csv")
print(data)

plt.scatter(data.enginesize,data.price)

X = data['enginesize'].values
y = data['price'].values

m_X = np.mean(X)
standard_X = np.std(X)
X_normalized = (X - m_X) / standard_X

def gradient_descent(m, b, L, X, y):
    m_gradient = 0
    b_gradient = 0
    n = len(X)  

    for i in range(n):
        x = X[i] 
        y_i = y[i]  
        
        
        m_gradient += (-2/n) * x * (y_i - (m * x + b))
        b_gradient += (-2/n) * (y_i - (m * x + b))

    m -= L * m_gradient
    b -= L * b_gradient

    return m, b

m = 0  
b = 0 
L = 0.001  
epochs = 1000  


for i in range(epochs):
    m, b = gradient_descent(m, b, L, X_normalized, y)

predicted_prices = m * X_normalized + b

plt.scatter(X, y, label='Actual data')
plt.plot(X, predicted_prices, color='red', label='Linear Regression')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.title('Linear Regression for Car Prices based on Engine Size')
plt.show()
