import numpy as np
import matplotlib.pyplot as plt

# Sample binary classification data
X = np.array([1, 2, 3, 4, 5])
y = np.array([0, 0, 0, 1, 1])
X = X.reshape(-1, 1)  # reshape for matrix operations

# Add bias (intercept term)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Initialize weights
theta = np.zeros(X_b.shape[1])
lr = 0.1
epochs = 1000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training
for _ in range(epochs):
    z = np.dot(X_b, theta)
    h = sigmoid(z)
    gradient = np.dot(X_b.T, (h - y)) / y.size
    theta -= lr * gradient

# Predictions
x_plot = np.linspace(0, 6, 100)
x_plot_b = np.c_[np.ones((100, 1)), x_plot.reshape(-1, 1)]
pred_probs = sigmoid(np.dot(x_plot_b, theta))

# Plot
plt.scatter(X, y, color="blue", label="Original data")
plt.plot(x_plot, pred_probs, color="green", label="Sigmoid curve")
plt.title("Logistic Regression from Scratch")
plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.show()
