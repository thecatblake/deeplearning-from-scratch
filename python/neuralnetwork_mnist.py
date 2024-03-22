import pickle

import numpy as np
from keras.datasets import mnist

from activation import sigmoid, softmax

(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_test = np.reshape(x_test, (-1,784))

with open("sample_weight.pkl", "rb") as f:
    network = pickle.load(f)

W1, W2, W3 = network["W1"], network["W2"], network["W3"]
b1, b2, b3 = network["b1"], network["b2"], network["b3"]

batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x_test), batch_size):
    a1 = np.dot(x_test[i:i+batch_size], W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    p = np.argmax(y, axis=1)

    accuracy_cnt += np.sum(p == t_test[i:i+batch_size])

print("Accuracy: " + str(float(accuracy_cnt) / len(x_test)))
