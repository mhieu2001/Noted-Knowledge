# Noted-Knowledge
## 1. What is Convolution?
Convolution is a mathematical operation which involves combining two functions to produce a third function. It essentially “blends” or “merges” two functions together. For instance, if you have two functions, f(x) and g(x), convolution can be employed to combine them and generate a new function, denoted as (f * g)(x). This new function captures the combined impact of the original functions, illustrating their interaction and overlap.

## 1.1 How Does a CNN Work?
Convolution in CNNs allows the network to automatically learn and extract visual features from images, model spatial relationships, handle variations in object position and appearance, and capture meaningful relationships within images. Important characteristics in images can be captured using CNNs, such as edges, corners, textures, and shapes.

In a CNN’s convolutional filter context, f(x) represents the input data, and g(x) represents the convolutional filter used in the network. The input data is a small matrix typically from an image or a feature map produced by a previous layer in the network, and the convolutional filter is a small matrix of weights that was learned in the training process. The convolutional filter essentially acts as a window that scans the input data to extract relevant features.

During convolution, the filter (g(x)) is slid over the input data (f(x)). At each position, the filter performs element-wise multiplication of the two matrices. The products are then summed to produce a single value which is the output of the convolution in that position. This process is repeated for all positions, which results in an “output feature map,” which is a two-dimensional representation of the activations produced by applying the convolutional filters to the input data.

## 1.2 Challenges of CNNs
Although CNNs are commonly employed in computer vision tasks, there are also some challenges, such as:

Computational cost: Training CNNs typically requires substantial computational resources, and training with large datasets of high-resolution images can be time-consuming. Training times can also be long, and specialized hardware (e.g., GPUs) is often used to accelerate computations.
Need for large datasets: CNNs need large amounts of labeled training data to generalize well and learn meaningful features. “Generalization” is a crucial aspect of machine learning and refers to a model’s ability to make reliable predictions for data it hasn’t been exposed to in the training process. A model that “generalizes well” has learned the relevant patterns and relationships during training, so it can effectively be used to make accurate predictions on new data. Insufficient training data for CNNs can lead to overfitting, which means the model does not generalize well and fails to make reliable predictions on new, unseen data.
Interpretability: CNNs are sometimes considered “black box models,” which means that their internal workings and decision-making processes are not easily interpretable by humans. The complex architectures and numerous parameters of CNNs make it difficult to interpret why specific predictions or decisions are made. This can raise concerns in domains where interpretability is crucial.

## 2. What is Convolution?
Convolution is a mathematical operation which involves combining two functions to produce a third function. It essentially “blends” or “merges” two functions together. For instance, if you have two functions, f(x) and g(x), convolution can be employed to combine them and generate a new function, denoted as (f * g)(x). This new function captures the combined impact of the original functions, illustrating their interaction and overlap.

## 2.1How Does a CNN Work?
Convolution in CNNs allows the network to automatically learn and extract visual features from images, model spatial relationships, handle variations in object position and appearance, and capture meaningful relationships within images. Important characteristics in images can be captured using CNNs, such as edges, corners, textures, and shapes.

In a CNN’s convolutional filter context, f(x) represents the input data, and g(x) represents the convolutional filter used in the network. The input data is a small matrix typically from an image or a feature map produced by a previous layer in the network, and the convolutional filter is a small matrix of weights that was learned in the training process. The convolutional filter essentially acts as a window that scans the input data to extract relevant features.

During convolution, the filter (g(x)) is slid over the input data (f(x)). At each position, the filter performs element-wise multiplication of the two matrices. The products are then summed to produce a single value which is the output of the convolution in that position. This process is repeated for all positions, which results in an “output feature map,” which is a two-dimensional representation of the activations produced by applying the convolutional filters to the input data.

## 2.2 Challenges of CNNs
Although CNNs are commonly employed in computer vision tasks, there are also some challenges, such as:

Computational cost: Training CNNs typically requires substantial computational resources, and training with large datasets of high-resolution images can be time-consuming. Training times can also be long, and specialized hardware (e.g., GPUs) is often used to accelerate computations.
Need for large datasets: CNNs need large amounts of labeled training data to generalize well and learn meaningful features. “Generalization” is a crucial aspect of machine learning and refers to a model’s ability to make reliable predictions for data it hasn’t been exposed to in the training process. A model that “generalizes well” has learned the relevant patterns and relationships during training, so it can effectively be used to make accurate predictions on new data. Insufficient training data for CNNs can lead to overfitting, which means the model does not generalize well and fails to make reliable predictions on new, unseen data.
Interpretability: CNNs are sometimes considered “black box models,” which means that their internal workings and decision-making processes are not easily interpretable by humans. The complex architectures and numerous parameters of CNNs make it difficult to interpret why specific predictions or decisions are made. This can raise concerns in domains where interpretability is crucial. 

## 3. Activation Function
An activation function is the function used by a node in a neural network to take the summed weighted input to the node and transform it into the output value. The output value can be fed into the next hidden layer or received from the output layer. For networks that engage in tasks of any significant complexity, the activation function needs to be non-linear.

## 3.1 Types of Activation Functions
There are a number of different types of activation functions that can be used:

Linear: The node returns what was provided as input, also known as the “identity” function.
Binary Step: The node either provides an output or not, depending on some threshold applied to the input.
Sigmoid: Takes the input and returns a value between 0 and 1. The more positive (larger) the input, the closer the value is to 1. The more negative (smaller) the input, the closer the value is to 0.
Tanh: Like the sigmoid function but produces values between -1 and 1.
Gaussian: Takes the input and distributes it in a bell curve with values between 0 and 1.
ReLU (Rectified Linear Unit): Represents the positive part of the input, returning 0 for values less than 0, and behaving like a linear function for positive values.
