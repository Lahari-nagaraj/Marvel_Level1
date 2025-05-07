# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset (Make sure the CSV is in the same folder)
df = pd.read_csv("Absenteeism_at_work.csv", delimiter=";")

# Display basic info
print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing Values in Dataset:")
print(df.isnull().sum())

# Fill missing values with column mean
df.fillna(df.mean(), inplace=True)

# Verify changes
print("\nAfter Handling Missing Values:")
print(df.isnull().sum())

# Select relevant columns for clustering
selected_columns = ["Age", "Service time", "Work load Average/day ", "Transportation expense"]
df_selected = df[selected_columns]

# Standardizing the data
df_selected = (df_selected - df_selected.mean()) / df_selected.std()

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_selected["Cluster"] = kmeans.fit_predict(df_selected)

# Scatter plot to visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_selected, x="Age", y="Work load Average/day ", hue="Cluster", palette="viridis")
plt.title("K-Means Clustering on Absenteeism Data")
plt.show()

# Pairplot to visualize distributions
sns.pairplot(df_selected, hue="Cluster", palette="viridis")
plt.show()

# Save the processed dataset with clusters
df_selected.to_csv("clustered_absenteeism.csv", index=False)
print("Clustered dataset saved as 'clustered_absenteeism.csv'")
