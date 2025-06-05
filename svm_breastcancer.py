# svm_breast_cancer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # 0 = malignant, 1 = benign

# Split data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Try kernel='linear' if needed
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Output Results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# (Optional) Visualize class distribution
sns.countplot(x='target', data=df)
plt.title("Tumor Type Distribution (0 = Malignant, 1 = Benign)")
plt.show()
