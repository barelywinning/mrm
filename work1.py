import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
car_price = pd.read_csv(r"C:\Users\adity\Downloads\CarPrices\CarPrice_Assignment.csv")
data = car_price.drop(["CarName","enginelocation", "fuelsystem"], axis=1)
columns_to_encode = ["fueltype","aspiration","doornumber","drivewheel","cylindernumber","carbody","enginetype"]
label_mappings = {}
for column in columns_to_encode:
    unique_categories = data[column].unique()
    label_mapping = {}
    for i, category in enumerate(unique_categories):
        label_mapping[category] = i
    label_mappings[column] = label_mapping
for column, label_mapping in label_mappings.items():
    data[column] = data[column].map(label_mapping)
print(data.describe())
X = data.drop("price", axis=1)
y = data["price"]
fmin = X.min(axis=0)
frange = X.max(axis=0) - X.min(axis=0)
X = (X - fmin) / frange
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
weights = np.random.randn(X_train.shape[1]) * 0.01
bias = 0.02
learning_rate = 0.1
iterations = 5000
alpha = 0.1
cost_history = []
def costFunction(y, predictions, weights, alpha):
    N = len(y)
    sq_error = (predictions - y) ** 2
    regularization_term = (alpha / (2 * N)) * np.sum(weights**2)
    return (1.0 / (2 * N)) * sq_error.sum() + regularization_term


for i in range(iterations):
    predictions = np.dot(X_train, weights) + bias
    error = predictions - y_train
    N = X_train.shape[0]
    dw = (1 / N) * np.dot(X_train.T, error) + (alpha / N) * weights
    db = np.sum(error) / N
    weights -= learning_rate * dw
    bias -= learning_rate * db
    current_cost = costFunction(y_train, predictions, weights, alpha)
    cost_history.append(current_cost)


predictions_test = np.dot(X_test, weights) + bias
y_mean = y_test.mean()
tss = ((y_test - y_mean) ** 2).sum()
rss = ((y_test - predictions_test) ** 2).sum()
r_squared = 1 - (rss / tss)

print(f'R-squared (R²) value: {r_squared:.4f}')


plt.figure(figsize=(10, 6))
plt.plot(range(iterations), cost_history)
plt.title('Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Cost Function Value')
plt.grid(True)
plt.show()
