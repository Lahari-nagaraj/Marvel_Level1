# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = datasets.load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target  # 0: Setosa, 1: Versicolor, 2: Virginica

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Define features (X) and target (y)
X = iris.data  # Features: sepal/petal length & width
y = iris.target  # Labels: 0, 1, or 2 (species)

# Split into train-test (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict species on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Visualize data distribution
sns.pairplot(df, hue='Species', diag_kind='hist')
plt.show()
