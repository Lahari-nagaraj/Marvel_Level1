# K-Nearest Neighbors on Iris Dataset (Human-style Version)
# Includes: From-scratch KNN, sklearn KNN, and 2D plot using PCA

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Euclidean Distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# KNN from Scratch
class MyKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        # Calculate distances to all training points
        distances = [euclidean_distance(x, point) for point in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        # Return the most common label
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(np.array(predictions) == y)

# Train and evaluate MyKNN
my_knn = MyKNN(k=5)
my_knn.fit(X_train, y_train)
custom_accuracy = my_knn.score(X_test, y_test)
print(f"🧠 Custom KNN Accuracy: {custom_accuracy * 100:.2f}%")

# Train and evaluate sklearn KNN
sklearn_knn = KNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(X_train, y_train)
sklearn_accuracy = sklearn_knn.score(X_test, y_test)
print(f"🤖 scikit-learn KNN Accuracy: {sklearn_accuracy * 100:.2f}%")

# PCA for 2D visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Predict all points using custom KNN for plotting
my_knn.fit(X_reduced, y)
custom_predictions = my_knn.predict(X_reduced)

# Plotting
plt.figure(figsize=(8, 6))
for i, label in enumerate(np.unique(y)):
    plt.scatter(
        X_reduced[np.array(custom_predictions) == label, 0],
        X_reduced[np.array(custom_predictions) == label, 1],
        label=f"Predicted {target_names[label]}",
        alpha=0.6
    )

plt.title("KNN Prediction (2D PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
