import random

import numpy as np
from torch.nn.functional import sigmoid
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    dataset = MNIST(root='./data', transform=lambda x: np.array(x).flatten(), download=True, train=is_train)
    mnist_data, mnist_labels = [], []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)


def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.b_hidden = np.random.randn(hidden_size)
        self.w_output = np.random.randn(hidden_size, output_size) * 0.01
        self.b_output = np.random.randn(output_size)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(z))

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_hidden_layer(self, X):
        z_hidden = np.dot(X, self.w_hidden) + self.b_hidden
        a_hidden = self.sigmoid(z_hidden)
        return a_hidden, z_hidden

    def forward_output_layer(self, a_hidden):
        z_output = np.dot(a_hidden, self.w_output) + self.b_output
        y_pred = self.softmax(z_output)
        return y_pred, z_output

    def backpropagation(self, X_batch, y_true, a_hidden, z_hidden, y_pred, learning_rate):
        delta_output = y_pred - y_true
        gradient_W_output = np.dot(a_hidden.T, delta_output)
        gradient_b_output = np.sum(delta_output, axis=0)

        delta_hidden = np.dot(delta_output, self.W_output.T)
        delta_hidden[z_hidden <= 0] = 0  # Derivata ReLU
        gradient_W_hidden = np.dot(X_batch.T, delta_hidden)
        gradient_b_hidden = np.sum(delta_hidden, axis=0)

        self.W_output -= learning_rate * gradient_W_output
        self.b_output -= learning_rate * gradient_b_output
        self.W_hidden -= learning_rate * gradient_W_hidden
        self.b_hidden -= learning_rate * gradient_b_hidden

    def train_model(self, train_X, train_Y, epochs=50, batch_size=100, learning_rate=0.01):
        for epoch in range(epochs):
            num_batches = train_X.shape[0] // batch_size
            for i in range(num_batches):
                X_batch = train_X[i * batch_size:(i + 1) * batch_size]
                y_batch = train_Y[i * batch_size:(i + 1) * batch_size]
                a_hidden, z_hidden = self.forward_hidden_layer(X_batch)
                y_pred, z_output = self.forward_output_layer(a_hidden)
                print(y_batch.shape)

                self.backpropagation(X_batch, y_batch, a_hidden, z_hidden, y_pred, learning_rate)

        accuracy = self.compute_accuracy(train_X, train_Y)
        print(f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {accuracy * 100:.2f}%")

    def compute_accuracy(self, X, y):
        a_hidden, _ = self.forward_hidden_layer(X)
        y_pred, _ = self.forward_output_layer(a_hidden)
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels)
        return accuracy


rn = NeuralNetwork(784, 100, 10)
train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)
train_X = train_X / 255.0
test_X = test_X / 255.0
train_Y_oh = one_hot_encode(train_Y, 10)
test_Y_oh = one_hot_encode(test_Y, 10)
rn.train_model(train_X, train_Y_oh)
