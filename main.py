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

# Parameters
n_neurons = [784, 128, 128, 10]

# Initialize weights and biases
weights = [np.random.normal(0, np.sqrt(1/n_neurons[i]), size=(n_neurons[i+1], n_neurons[i])) for i in range(len(n_neurons)-1)]
biases = [np.random.normal(0, 0.001, size=(n_neurons[i+1], 1)) for i in range(len(n_neurons)-1)]

# Activation functions
activation_functions = [relu, relu, relu, softmax]
dactivation_functions = [drelu, drelu, drelu, dsoftmax]

# Predict function
def predict(x, weights, biases):
    pre_activations = [x]
    for i in range(len(n_neurons)-1):
        pre_activation = np.dot(weights[i], x) + biases[i]
        pre_activations.append(pre_activation)
        x = activation_functions[i](pre_activation)
    return x, pre_activations

# Backpropagation function
def backpropagation(y_pred, y_actual, pre_activations, weights, biases, alpha):
    errors = [y_pred - y_actual]
    wgrads, bgrads = [], []

    for l in range(len(n_neurons)-1, 0, -1):
        activation_function = activation_functions[l-1]
        dactivation_function = dactivation_functions[l-1]
        
        if l != len(n_neurons)-1:
            error_l = (weights[l].T @ errors[len(n_neurons)-2-l]) * dactivation_function(pre_activations[l])
            errors.append(error_l)

        wgrads.append(np.outer(errors[len(n_neurons)-1-l], activation_function(pre_activations[l-1])))
        bgrads.append(errors[len(n_neurons)-1-l])

    wgrads.reverse()
    bgrads.reverse()

    for i in range(len(n_neurons)-1):
        weights[i] -= alpha * wgrads[i]
        biases[i] -= alpha * bgrads[i]

    return weights, biases

def convert_to_class(y_pred):
    return np.argmax(y_pred)

# Save function using pickle
def save_weights_and_biases(weights, biases, epoch):
    with open(f"weights_epoch_{epoch}.pkl", 'wb') as f:
        pickle.dump(weights, f)
    with open(f"biases_epoch_{epoch}.pkl", 'wb') as f:
        pickle.dump(biases, f)
    print(f"Weights and biases saved for epoch {epoch}.")
# Main Training Loop
import mnist
from tqdm import tqdm

mnist.datasets_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

num_train_examples = 10000  # Use a subset of training data
num_test_examples = test_images.shape[0]  # Use all test examples

epochs = 20
alpha = 0.01  # Learning rate

for epoch in range(epochs):
    print(f"Epoch {epoch}/{epochs}:")
    correct_train = 0

    # Training Phase
    for i in tqdm(range(num_train_examples), desc="Training Progress", unit="sample"):
        # Training data
        x = train_images[i].flatten()[:, np.newaxis] / 255
        y_actual = np.eye(10)[train_labels[i]][:, np.newaxis]

        y_pred, pre_activations = predict(x, weights, biases)

        if convert_to_class(y_pred) == convert_to_class(y_actual):
            correct_train += 1

        weights, biases = backpropagation(y_pred, y_actual, pre_activations, weights, biases, alpha)

    train_accuracy = correct_train / num_train_examples * 100
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    # Testing Phase (No Backpropagation)
    correct_test = 0
    for i in tqdm(range(num_test_examples), desc="Testing Progress", unit="sample"):
        x = test_images[i].flatten()[:, np.newaxis] / 255
        y_actual = np.eye(10)[test_labels[i]][:, np.newaxis]

        y_pred, _ = predict(x, weights, biases)  # No need to store pre-activations

        if convert_to_class(y_pred) == convert_to_class(y_actual):
            correct_test += 1

    test_accuracy = correct_test / num_test_examples * 100
    print(f"Testing Accuracy: {test_accuracy:.2f}%\n")

# Save weights and biases after training
save_weights_and_biases(weights, biases, epochs)