from torch import nn
import torch


class CNNNetwork(nn.Module):

    def __init__(self, out_neurons, input_shape, layers=None, kernel=3, stride=1, padding=2):
        super().__init__()
        assert len(input_shape) == 3
        c, h, w = input_shape
        self.layers = [c] + layers if layers else [c, 16, 32, 64, 128]
        self.out_neurons = out_neurons
        self.verbose = True
        # 4 conv blocks / flatten / linear / softmax
        self.conv_layers = []
        self.kernel_size = kernel
        for i in range(len(self.layers) - 1):
            subm = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.layers[i],
                    out_channels=self.layers[i+1],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.add_module(f'conv_layer{i}', subm)
            self.conv_layers.append(subm)

        self.flatten = nn.Flatten()
        for i in range(len(self.layers)-1):
            layer = self.conv_layers[i][0]
            h = (h + 2*layer.padding[0] - layer.dilation[0] *
                 (layer.kernel_size[0] - 1) - 1)/layer.stride[0] + 1
            w = (w + 2*layer.padding[1] - layer.dilation[1] *
                 (layer.kernel_size[1] - 1) - 1)/layer.stride[1] + 1
        linear_in = int(self.layers[-1]*h*w)
        print(linear_in)
        #self.linear = nn.Linear(linear_in, self.out_neurons)
        #self.linear = nn.Linear(3456, self.out_neurons)
        self.linear = None
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = input_data
        if self.verbose:
            print(x.shape)
        for conv in self.conv_layers:
            x = conv(x)
            if self.verbose:
                print(x.shape)

        x = self.flatten(x)
        if self.linear is None:
            self.linear = nn.Linear(x.shape[1], self.out_neurons)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        self.verbose = False
        return predictions