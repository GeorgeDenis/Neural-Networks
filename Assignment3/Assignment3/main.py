import random
import numpy as np
from torchvision.datasets import MNIST


def print_progress(bars):
    print("| ", end="")
    for i in range(bars):
        print("|", end="")
    if 100 - bars > 0:
        for i in range(100 - bars):
            print(" ", end="")
        print(" |", end="")
    print()


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
        return 1.0 / (1.0 + np.exp(-z))

    def softmax(self, z):
        max_z = np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z) / max_z
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_hidden_layer(self, X):
        z_hidden = np.dot(X, self.w_hidden) + self.b_hidden
        y_hidden = self.sigmoid(z_hidden)
        return y_hidden, z_hidden

    def forward_output_layer(self, a_hidden):
        z_output = np.dot(a_hidden, self.w_output) + self.b_output
        y_hidden = self.softmax(z_output)
        return y_hidden, z_output

    def sigmoid_derivative(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)

    def backpropagation(self, X_batch, y_true, a_hidden, z_hidden, y_pred, learning_rate):
        delta_output = y_pred - y_true

        gradient_W_output = np.dot(a_hidden.T, delta_output)
        gradient_b_output = np.sum(delta_output, axis=0)

        delta_hidden = np.dot(delta_output, self.w_output.T) * self.sigmoid_derivative(z_hidden)
        gradient_W_hidden = np.dot(X_batch.T, delta_hidden)
        gradient_b_hidden = np.sum(delta_hidden, axis=0)

        self.w_output -= learning_rate * gradient_W_output
        self.b_output -= learning_rate * gradient_b_output
        self.w_hidden -= learning_rate * gradient_W_hidden
        self.b_hidden -= learning_rate * gradient_b_hidden

    def train_model(self, train_X, train_Y, test_X, test_Y, epochs=100, batch_size=100, learning_rate=0.01):
        for epoch in range(epochs):
            num_batches = train_X.shape[0] // batch_size
            for i in range(num_batches):
                X_batch = train_X[i * batch_size:(i + 1) * batch_size]
                y_batch = train_Y[i * batch_size:(i + 1) * batch_size]
                y_hidden, z_hidden = self.forward_hidden_layer(X_batch)
                y_pred, z_output = self.forward_output_layer(y_hidden)

                self.backpropagation(X_batch, y_batch, y_hidden, z_hidden, y_pred, learning_rate)

            train_accuracy = self.compute_accuracy(train_X, train_Y)
            val_accuracy = self.compute_accuracy(test_X, test_Y)
            if epoch % 10 == 0 or (epoch + 1) == 100:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Training Accuracy: {train_accuracy * 100:.2f}% - Validation Accuracy: {val_accuracy * 100:.2f}%")
                print_progress(epoch)

    def compute_accuracy(self, X, y):
        y_hidden, _ = self.forward_hidden_layer(X)
        y_pred, _ = self.forward_output_layer(y_hidden)
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
rn.train_model(train_X, train_Y_oh, test_X, test_Y_oh)
