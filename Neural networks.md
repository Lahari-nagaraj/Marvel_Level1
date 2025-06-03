# Neural Networks

If you've ever wondered how AI actually “thinks” or learns, then neural networks are where that magic happens. Inspired by the human brain, neural networks are the backbone of deep learning. They learn patterns from data and can perform tasks like classifying images, translating languages, or predicting stock prices. In Machine Learning (ML), neural networks are a core concept used when traditional algorithms just don’t cut it — especially with large, messy, or high-dimensional data like images, voice, or even handwriting.

They fall under Supervised Learning (mostly), and what makes them cool is their ability to automatically extract features from raw data without needing you to manually design every rule. So instead of writing a program to detect spam, you train a model on examples of spam and not-spam, and the neural network learns the pattern.

## How Do Neural Networks Work?
A neural network is a collection of interconnected nodes (neurons) organized in layers:

Input layer: Receives the features (like pixel values or text embeddings)
Hidden layers: Where the actual “learning” happens.<br>
Output layer: Produces the final prediction

### How Do They Actually Learn?
This is where loss functions and backpropagation come in:<br>
Loss function measures how wrong the prediction was.
Backpropagation uses gradients (via calculus) to update weights to reduce that loss.<br>
Optimizer (like SGD or Adam) helps minimize the loss efficiently.

This process repeats for many iterations until the model learns well.


### What is Backpropagation?
This is the learning part of neural networks:

It calculates how wrong the model was using a loss function (like Mean Squared Error or Cross-Entropy).
Then it computes gradients (using calculus) and adjusts the weights via gradient descent.
This process repeats over multiple iterations until the model gets better.

Every neuron computes:
Z = w1*x1 + w2*x2 + ... + wn*xn + b
A = activation(Z)<br>
where:
x = input features,w = weights (learned during training),b = bias term
activation() = non-linear function like ReLU, sigmoid, tanh, etc.
The activation function is what helps the model learn complex patterns, not just straight lines.


## Types of Neural Networks

### 1. ANN (Artificial Neural Network)

ANN is the basic form of neural network inspired by how our brain neurons connect. It consists of layers of neurons where each neuron is connected to all neurons in the next layer (fully connected). It works well for structured data like tabular datasets.

ANN learns patterns from data by adjusting weights during training to make predictions like classification or regression.

##### Advantages:
Simple and easy to implement<br>
Works well on structured/tabular data<br>
Good for general-purpose problems

##### Disadvantages:
Not good with spatial or sequential data<br>
Can overfit easily if network is too large<br>
Requires more data preprocessing

Example code snippet :

from tensorflow.keras.models import Sequential<br>
from tensorflow.keras.layers import Dense<br>

model = Sequential()<br>
model.add(Dense(32, activation='relu', input_shape=(input_dim,)))<br>
model.add(Dense(1, activation='sigmoid'))<br>
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])<br>
model.fit(X_train, y_train, epochs=10)<br>

## 2. CNN (Convolutional Neural Network)

CNN is a specialized network designed to work with image and spatial data. Instead of fully connected layers, CNN uses convolutional layers to scan parts of images and detect features like edges and shapes.

It extracts spatial features from images and is excellent for image recognition, object detection, and computer vision tasks.

##### Advantages:
Automatically detects important features from images<br>
Reduces number of parameters via parameter sharing (filters)<br>
Handles spatial relationships well<br>

##### Disadvantages:
Requires lots of computational power<br>
Not suitable for sequence or time-series data<br>
Needs large datasets for good performance

Example code snippet:

from tensorflow.keras.models import Sequential<br>
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense<br>

model = Sequential()<br>
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)))<br>
model.add(MaxPooling2D((2,2)))<br>
model.add(Flatten())<br>
model.add(Dense(64, activation='relu'))<br>
model.add(Dense(1, activation='sigmoid'))<br>
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])<br>
model.fit(X_train, y_train, epochs=10)<br>



## 3. RNN (Recurrent Neural Network)

RNN is designed to handle sequence data by having loops (recurrent connections) that remember information from previous time steps. This memory helps process sequences like text, speech, or time series.


It models dependencies over time, making it great for language modeling, speech recognition, and time-series forecasting.

##### Advantages:
Can process sequences of variable length<br>
Captures temporal dependencies in data<br>
Useful for tasks where order matters

##### Disadvantages:
Can suffer from vanishing gradients (hard to learn long dependencies)<br>
Training can be slow<br>
More complex than ANN or CNN

Example code snippet :

from tensorflow.keras.models import Sequential<br>
from tensorflow.keras.layers import LSTM, Dense<br>

model = Sequential()<br>
model.add(LSTM(50, input_shape=(timesteps, features)))<br>
model.add(Dense(1, activation='sigmoid'))<br>
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])<br>
model.fit(X_train, y_train, epochs=10)<br>

| ANN                               | CNN                               | RNN                                    |
| --------------------------------- | --------------------------------- | -------------------------------------- |
| Tabular / vector data             | Image / grid data                 | Sequence / time series                 |
|  No parameter sharing            |  Yes (filters shared)            |  Yes (over time steps)                |
| Fixed input length                | Fixed input length                | Variable input length                  |
|  No recurrent connections        |  No recurrent connections        | Has recurrent connections            |
|  Ignores spatial relationships   |  Captures spatial relationships  |  Ignores spatial relationships        |
| Good for simple classification    | Good for image/video recognition  | Good for text, speech, predictions     |
| Works on generic tabular data     | Works on visual/spatial data      | Works on time-dependent data           |
| Low noise tolerance               | High noise tolerance (pooling)    | Medium noise tolerance                 |
| Medium accuracy potential         | High accuracy potential           | High accuracy with LSTM/GRU            |
|  Poor for time series prediction |  Poor for time series prediction | Excellent for time series prediction |
