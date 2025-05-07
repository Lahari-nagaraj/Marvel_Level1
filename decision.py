import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Create a dataset with Company, Job Type, Degree, and Salary > 10K (binary classification)
np.random.seed(42)

# Create a small dataset with the following features
data = {
    'Company': np.random.choice(['Tech', 'Finance', 'Healthcare'], size=100),
    'Job Type': np.random.choice(['Full-time', 'Part-time', 'Contract'], size=100),
    'Degree': np.random.choice(['Bachelors', 'Masters', 'PhD'], size=100),
    'Salary': np.random.randint(8000, 120000, size=100)  # Random salary between 8K and 120K
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Step 2: Create a binary target variable (Salary > 10K or not)
df['Above 10K'] = np.where(df['Salary'] > 10000, 1, 0)  # 1 if Salary > 10K, else 0

# Step 3: Encode categorical variables (Company, Job Type, Degree)
label_encoder = LabelEncoder()
df['Company'] = label_encoder.fit_transform(df['Company'])
df['Job Type'] = label_encoder.fit_transform(df['Job Type'])
df['Degree'] = label_encoder.fit_transform(df['Degree'])

# Step 4: Prepare feature matrix (X) and target vector (y)
X = df[['Company', 'Job Type', 'Degree']]  # Features
y = df['Above 10K']  # Target variable

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Step 7: Evaluate the model (accuracy on test data)
accuracy = classifier.score(X_test, y_test)
print(f'Accuracy of the Decision Tree Classifier: {accuracy * 100:.2f}%')

# Step 8: Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(classifier, filled=True, feature_names=['Company', 'Job Type', 'Degree'], class_names=['Salary <= 10K', 'Salary > 10K'], rounded=True)
plt.title("Decision Tree for Salary Prediction")
plt.show()
