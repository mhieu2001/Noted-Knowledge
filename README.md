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

## 2. Recurrent Neural Network
A Recurrent Neural Network (RNN) is a type of neural network that stores and re-uses the outputs from previous steps as additional inputs in the current step. RNNs process sequential data in groups of timesteps which represent a segment of the sequence. These timesteps could represent words from a sentence or frames from an audio clip. The network uses the output from previous timesteps which are stored in a “hidden state” or “hidden variable” to compute the output for the current timestep.

RNNs are used in cases where the sequence of data is important to the predicted output. In sequential or time-based data sets, RNNs use previous outputs as “memories” or “states”, and use that to predict the output in the sequence.

## 2.1 How Does an RNN Work?
The training data is split into time-based sequences. They are then vector-encoded and passed into the RNN as inputs. The RNN is initialized with random weights and biases which remain the same through all timesteps. The network iterates over the sequence to compute the output or hidden state for each timestep.

The current hidden state is computed by passing the input for the current time-step, the previous hidden state, and the weights and biases for the sequence into a layer with an activation function of choice. The hidden state is then returned as an output and passed to the next timestep. The process is repeated through the rest of the timesteps. Depending on the task, the final output can either be returned as a single vector or can be a combination of output vectors returned in each timestep. For the first timestep, the previous hidden state can be initialized as a zero vector of size dependent on the intended output size.

A loss function is used to compare the expected result and actual result returned by the model. Using the loss calculated, the loss gradient for each timestep is computed using Backpropagation Through Time (BPTT) and is used to update the weights in order to reduce the loss. Repeating this process trains the network until the expected result is produced.

A common issue when training an RNN is the vanishing gradient problem. When backpropagating and updating the weights, the error gradient becomes smaller and smaller as it approaches the earlier inputs. As a result, the earlier inputs will have less of an impact on training process. To rectify this, networks like Long Short Term Memory (LSTM) and Gated Recurrent Unit (GRU) are often used. Although out of scope for the purpose of this doc, LSTM and GRU are both worth looking into, as well as another common problem that can come up - the exploding gradient. 

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

## 4. What is Backpropagation?
Backpropagation, short for “backward propagation of errors,” is a supervised learning algorithm that calculates the gradient of the loss function with respect to the network’s weights. It allows the determination of how much each weight contributes to the overall error or loss of the network’s predictions.

## 4.1 How is Backpropagation Used in Artificial Neural Networks?
Artificial Neural Networks consist of interconnected nodes, called neurons, organized in layers. These layers include an input layer, one or more hidden layers, and an output layer. Backpropagation is used to adjust the weights and biases of the connections between these neurons.

The backpropagation algorithm can be summarized in the following steps:

- Forward Pass: During the forward pass, input data is propagated through the network, layer by layer, until the output layer is reached. Each neuron in a layer receives inputs from the previous layer, calculates a weighted sum, applies an activation function, and passes the result to the next layer.

- Calculating the Error: After the forward pass, the network’s output is compared to the expected output using a loss function. The error is the discrepancy between the predicted output and the desired output.

- Backward Pass: In the backward pass, the error is propagated back through the network, starting from the output layer towards the input layer. This is where backpropagation gets its name. The error is assigned to each neuron in proportion to its contribution to the overall error.

- Weight and Bias Updates: Using the calculated errors, the algorithm adjusts the weights and biases of the network’s connections. This adjustment is done iteratively, typically using an optimization algorithm like gradient descent, which minimizes the error by updating the weights in the direction opposite to the gradient.

- Repeat: Steps 1 to 4 are repeated for a fixed number of iterations or until a convergence criterion is met. The network gradually learns to minimize the error and improve its predictions.

## 4.2 Benefits and Importance of Backpropagation
Backpropagation is a fundamental technique in training artificial neural networks due to its numerous benefits:

- Efficient Training: Backpropagation allows neural networks to efficiently learn complex relationships in data, making them capable of solving complex problems.

- Universal Approximators: ANNs with backpropagation have the ability to approximate any continuous function, given enough neurons and training data.

- Generalization: By adjusting weights and biases, backpropagation enables the neural network to generalize from training data to make accurate predictions on unseen data.

- Adaptability: Backpropagation allows neural networks to adapt and improve their performance over time, making them suitable for tasks that involve changing environments or evolving data patterns.

- Deep Learning: Backpropagation forms the basis of deep learning, enabling the training of deep neural networks with many layers and millions of parameters.
