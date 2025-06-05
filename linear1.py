import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 4, 2, 5, 6])

# Initialize parameters
m = 0
c = 0
lr = 0.01
epochs = 1000
n = len(X)

# Training using Gradient Descent
for _ in range(epochs):
    y_pred = m * X + c
    error = y - y_pred
    m_grad = (-2/n) * sum(X * error)
    c_grad = (-2/n) * sum(error)
    m -= lr * m_grad
    c -= lr * c_grad

# Final prediction
y_final = m * X + c

# Plot
plt.scatter(X, y, color="blue", label="Original data")
plt.plot(X, y_final, color="red", label="Fitted line")
plt.title("Linear Regression from Scratch")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
