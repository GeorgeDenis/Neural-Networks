import torch


class MyNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 1),
            torch.nn.ReLU()
        )

    def forward(self, input):
        return self.layers(input)


neural = MyNN()
print(neural.layers[0].weight)
input = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [2.0, 4.0, 6.0]])
print(neural.forward(input))
