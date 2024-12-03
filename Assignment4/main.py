import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torch import device

image_to_tensor = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='/mnist-dataset', train=True, transform=image_to_tensor, download=True)
test_data = datasets.MNIST(root='/mnist-dataset', train=False, transform=image_to_tensor, download=True)

loaders = {
    'train': DataLoader(train_data, batch_size=64, shuffle=True),
    'test': DataLoader(test_data, batch_size=500, shuffle=True)
}


class MyNN(nn.Module):
    def __init__(self):
        super(MyNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 10, bias=False)
        )

    def forward(self, x):
        return self.layers(x)


def train_model(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(50):
        print(f"Epoch {epoch + 1}")

        model.train()

        for batch_idx, (inputs, labels) in enumerate(tqdm(loaders['train'])):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.reshape(inputs.shape[0], -1)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        train_acc = check_accuracy(loaders['train'], model)
        test_acc = check_accuracy(loaders['test'], model)
        print(f"Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%")

        scheduler.step()


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    model.train()
    return num_correct.item() / num_samples


def generate_submission(model, test_data):
    model.eval()
    data = {"ID": [], "target": []}

    with torch.no_grad():
        for idx, (image, _) in enumerate(test_data):
            image = image.to(device)
            image = image.reshape(1, -1)
            output = model(image)
            _, prediction = output.max(1)
            data["ID"].append(idx)
            data["target"].append(prediction.item())

    df = pd.DataFrame(data)
    df.to_csv("submission.csv", index=False)
    print("Saved in submission.csv!")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyNN().to(device)
train_model(model)
print(check_accuracy(loaders['train'], model))
generate_submission(model, test_data)
