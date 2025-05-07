# Task1: HelloWorld for AIML


This marked the beginning of my journey into AI and machine learning. Being new to the domain, this task gave me a hands-on introduction to how regression models operate. I implemented a Linear Regression model using the California Housing dataset to predict house prices based on various features. I also worked on a Logistic Regression model using the Iris dataset to classify different flower species. Through these exercises, I gained a clearer understanding of concepts like data preprocessing, model training, prediction, and performance evaluation.

githublink
githublink
image
image

# Task2 : Matplotlib and Data Visualisation

In this task, I got to understand what data visualization really means and how powerful it can be in exploring and interpreting data. I learned how Matplotlib works as a core plotting library and how it integrates with Seaborn for enhanced visualizations. Through this, I explored various plot types such as line plots, scatter and bubble plots (using the Iris dataset), bar charts (simple, grouped, and stacked), histograms, pie charts and other plot types. I worked with the Absenteeism at Work dataset, performed basic preprocessing, and applied K-Means clustering to visualize groupings in the data. This task gave me a clear understanding of how to represent multivariate data and introduced me to the basics of clustering and unsupervised learning.

githublink
image
image

# Task 3 - Numpy

This task gave me a great start with NumPy and helped me understand what it truly is—a powerful numerical computing library widely used in data science and machine learning. I explored fundamental operations such as array creation, reshaping, and repetition using the np.tile() method to repeat a small array across multiple dimensions. I also practiced generating arrays with np.arange(), sorting random values, and retrieving the indices of sorted elements. To reinforce my learning, I created a small Jupyter Notebook documenting key methods and syntax for quick reference in future projects.

github link
image


# Task 4 - Metrics and Performance Evaluation


This task helped me understand how we measure the effectiveness of machine learning models through various performance metrics. I explored both regression and classification tasks by implementing appropriate models and evaluating them using widely accepted metrics.

### Regression Metrics
I used the California Housing dataset for this part. After preprocessing the data (train-test split and feature scaling), I implemented two models:

Linear Regression &
Random Forest Regressor

To evaluate their performance, I used the following metrics:

MAE (Mean Absolute Error): Measures average magnitude of errors without considering direction.

MSE (Mean Squared Error): Penalizes larger errors more than MAE.

RMSE (Root Mean Squared Error): Square root of MSE, helps interpret error in original units.

R² Score (Coefficient of Determination): Indicates how well the model explains the variability of the target variable.


### Classification Metrics

For the classification part, I worked with the Breast Cancer dataset. I trained:
Logistic Regression,
Random Forest Classifier

To evaluate these models, I used:

Accuracy Score: Proportion of correctly predicted labels.

Classification Report: Provided precision, recall, and F1-score for each class.

Confusion Matrix: Visualized via a heatmap to identify false positives and false negatives.

These evaluations helped me compare not only the raw accuracy of the models, but also their ability to generalize and avoid critical errors in binary classification tasks.

# Task 5: Data Visualization for Exploratory Data Analysis

In this task, I explored Plotly, a powerful and interactive data visualization library. I learned how easy it is to create dynamic, web-based plots with minimal code, and how Plotly stands out from static tools like Matplotlib by allowing real-time interaction and exploration. Using the Gapminder dataset, I created a scatter plot to study the relationship between GDP per capita and life expectancy, a pie chart to show continent-wise country distribution, and a time series plot for stock prices. These visualizations helped me understand data patterns more intuitively and made the analysis more engaging.


# Task 6: An introduction to Decision Trees


A Decision Tree is a supervised learning algorithm used for classification and regression. It splits data into branches based on feature conditions, making decisions in a tree-like structure. In this task, I created a synthetic dataset with features like Company, Job Type, Degree, and Salary. I converted categorical data using label encoding and defined the target as whether Salary > 10K. Using scikit-learn, I trained a Decision Tree Classifier, evaluated its accuracy, and visualized the model to understand how different features influenced the outcome.








