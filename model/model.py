import numpy as np
import pickle

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def drelu(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def dsoftmax(x):
    s = softmax(x)
    return np.diagflat(s) - np.outer(s, s)

# Loss Functions
def cross_entropy(y_pred, y_actual, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1.0)
    return -1 * np.sum(y_actual * np.log(y_pred))

# Load function using pickle
def load_weights_and_biases(path, i):
    with open(f"{path}/weights_epoch_{i}.pkl", 'rb') as f:
        weights = pickle.load(f)
    with open(f"{path}/biases_epoch_{i}.pkl", 'rb') as f:
        biases = pickle.load(f)
    return weights, biases

def convert_to_class(y_pred):
    return np.argmax(y_pred)

# Layers
n_neurons = [784, 128, 128, 10]

# Activation functions
activation_functions = [relu, relu, relu, softmax]
dactivation_functions = [drelu, drelu, drelu, dsoftmax]

# Predict function
def predict(x, weights, biases):
    for i in range(len(n_neurons)-1):
        pre_activation = np.dot(weights[i], x) + biases[i]
        x = activation_functions[i](pre_activation)
    return x