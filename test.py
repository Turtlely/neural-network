import mnist
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model.model import *

# Import dataset
mnist.datasets_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

i=2006

test_image = mnist.test_images()[i]
test_label = mnist.test_labels()[i]

x = test_image.flatten()[:, np.newaxis] / 255
y_actual = np.eye(10)[test_label][:, np.newaxis]

# Parameters
weights, biases = load_weights_and_biases("./model", 20)

y_pred = predict(x, weights, biases)

# Print prediction and actual label
print("Predicted: ", convert_to_class(y_pred))
print("Actual: ", test_label)

# Display the image of the number
plt.imshow(test_image, cmap="gray")  # Reshape if necessary
plt.title(f"Actual: {test_label}, Predicted: {convert_to_class(y_pred)}")
plt.show()