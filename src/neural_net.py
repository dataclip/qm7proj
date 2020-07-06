from torch import nn, relu


class MLPNet(nn.Module):
    def __init__(self, x_inp_shape):
        super().__init__()
        self.fc1 = nn.Linear(x_inp_shape, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.fc3(x)
        x = x * 223.9345 - 1538.0376 # get y std dev and mean
        return x