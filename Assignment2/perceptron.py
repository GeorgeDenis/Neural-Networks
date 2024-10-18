import numpy as np
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


def initialize_parameters(input_size, output_size):
    W = np.random.randn(input_size, output_size) * 0.01
    b = np.zeros((output_size,))
    return W, b


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_propagation(X, W, b):
    z = np.dot(X, W) + b
    y_pred = softmax(z)
    return y_pred


def compute_cross_entropy(y_pred, y):
    return -np.sum(y * np.log(y_pred + 1e-8))


def backward_propagation(X, y_pred, y, W, b, learning_rate):
    dz = y - y_pred
    dW = np.dot(X.T, dz)
    db = np.sum(dz, axis=0)

    W += learning_rate * dW
    b += learning_rate * db

    return W, b

def train_model(train_X, train_Y, test_X, test_Y, W, b, epochs=50, batch_size=100, learning_rate=0.01):
    for epoch in range(epochs):
        num_batches = train_X.shape[0] // batch_size
        for i in range(num_batches):
            X_batch = train_X[i * batch_size:(i + 1) * batch_size]
            y_batch = train_Y[i * batch_size:(i + 1) * batch_size]

            y_pred = forward_propagation(X_batch, W, b)

            loss = compute_cross_entropy(y_pred, y_batch)

            W, b = backward_propagation(X_batch, y_pred, y_batch, W, b, learning_rate)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')

    return W, b


def predict(X, W, b):
    y_hat = forward_propagation(X, W, b)
    return np.argmax(y_hat, axis=1)

def evaluate_accuracy(X, y, W, b):
    predictions = predict(X, W, b)
    labels = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)


train_X = train_X / 255.0
test_X = test_X / 255.0
train_Y_oh = one_hot_encode(train_Y, 10)
test_Y_oh = one_hot_encode(test_Y, 10)
print(train_Y_oh)

W, b = initialize_parameters(784,10)
W, b = train_model(train_X, train_Y_oh, test_X, test_Y_oh, W, b, epochs=50, batch_size=100)
test_accuracy = evaluate_accuracy(test_X, test_Y_oh, W, b)
print(f'Test Accuracy: {test_accuracy}%')

test_instance = test_X[0]
test_instance = test_instance.reshape(1, 784)

predicted_label = predict(test_instance, W, b)

print(f'Predicted label: {predicted_label}')
print(f'True label: {test_Y[0]}')