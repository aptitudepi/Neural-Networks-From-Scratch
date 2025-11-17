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
        self.output = np.dot(inputs, self.weights) + self.biases

class activate_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class activate_Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        base = np.sum(exp_val, axis=1, keepdims=True)
        probs = exp_val / base
        self.output = probs

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
        
X, y = spiral_data(samples = 100, classes=3)
layer1 = layer_dense(2, 3)
activation1 = activate_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output[:5])
activation2 = activate_Softmax()
activation2.forward(activation1.output)
print(activation2.output[:5])

loss = Categorical_CrossEntropy().calculate(activation2.output, y)
print(f"loss: {loss}")

predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)
print(f"accuracy: {accuracy}")
