import numpy as np
import nnfs
from nnfs.datasets import spiral_data
class neuron:
    def __init__(self, weight: np.array, bias: float):
        self.weight = weight
        self.bias = bias

    def activate(self, input: np.array) -> int:
        return np.dot(self.weight, input) + self.bias


class layer:
    def __init__(self, weights: np.array, biases: np.array):
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        
    def activate(self, inputs):
        return np.dot(inputs, self.weights.T) + self.biases

class layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class activate_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.output <= 0] = 0
        
class activate_Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        base = np.sum(exp_val, axis=1, keepdims=True)
        probs = exp_val / base
        self.output = probs
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        
        data_loss = np.mean(sample_losses)
        return data_loss

class Categorical_CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis = 1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Softmax_Categorical_CrossEntropy:
    def __init__(self):
        self.activation = activate_Softmax()
        self.loss = Categorical_CrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    def __init__(self, learning_rate=0.5):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

# Create input dataset
X, y = spiral_data(samples = 100, classes=3)

# Define layers + functions
layer1 = layer_dense(2, 64)
activation1 = activate_ReLU()
layer2 = layer_dense(64, 3)
loss_activation = Softmax_Categorical_CrossEntropy()
optimizer = Optimizer_SGD()

for epoch in range(10001):
    # Forward pass
    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    loss = loss_activation.forward(layer2.output, y)

    # # Print fp results
    # print(loss_activation.output[:5])
    # print(f"loss: {loss}")
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    if not epoch % 100:
        print(f"epoch: {epoch} acc: {accuracy} loss: {loss}")
    # Backward pass
    loss_backward = loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # print(layer1.dweights)
    # print(layer1.dbiases)
    # print(layer2.dweights)
    # print(layer2.dbiases)

    # Update weights and biases
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)