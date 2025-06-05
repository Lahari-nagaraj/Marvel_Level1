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


# Task 7:  K- Nearest Neighbor Algorithm

 I explored the K-Nearest Neighbors (KNN) algorithm, a simple yet powerful supervised learning technique. KNN works by identifying the ‘k’ nearest data points to a new input and classifying it based on majority voting. It’s a lazy learning algorithm, meaning it doesn’t require model training and makes predictions directly using the training data.

I implemented KNN using scikit-learn’s KNeighborsClassifier on datasets like Iris, Wine, and Breast Cancer. I experimented with different k values and distance metrics to observe their effect on performance. I also learned the importance of feature scaling since KNN is distance-based.

To understand the algorithm better, I implemented KNN from scratch in Python. This helped me clearly see how distances are calculated, neighbors selected, and predictions made. I compared the results of my custom implementation with scikit-learn’s and found them closely matching, though the library version was faster.

Overall, this task helped me understand the working, strengths, and limitations of KNN.


gthub
inage

# Task 8: An elementary step towards understanding Neural Networks

While working on the neural networks task, I explored different types like ANN and CNN through hands-on coding. I understood how layers, weights, and activation functions contribute to the model’s predictions, and how backpropagation helps in adjusting the weights using gradients. Implementing small models helped me grasp the role of each layer clearly. Later, when learning about Large Language Models like GPT-4, I explored how transformers, self-attention, and token embeddings power language understanding. Though complex, breaking it down step by step helped me connect neural networks to how large models like GPT actually work.

blog
blog
image
image

# Task 9: Mathematics behind machine learning

I explored the mathematical foundations behind machine learning by working on curve fitting and Fourier transforms. For curve fitting, I used Desmos to visualize and model a function of my choice. I experimented with polynomial functions and learned how changing the degree of the polynomial affected the fit of the curve to the data points. This hands-on exercise helped me understand how curve fitting plays a crucial role in regression problems, where the goal is to find a function that best approximates the relationship between variables. I also learned how overfitting can occur with higher-degree polynomials and the importance of balancing complexity and generalization.













