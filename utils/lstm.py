from torch import nn
import torch
import torch.nn.functional as F


class LSTMNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, lstm_layers, linear_layers, h_activ=F.relu, out_activ=nn.Softmax(dim=1)):

        super(LSTMNetwork, self).__init__()

        self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True
            )

        linear_dims = [hidden_dim] + linear_layers
        self.num_linear = len(linear_layers)
        self.linear_layers = nn.ModuleList()
        for index in range(self.num_linear):
            layer = nn.Linear(linear_dims[index], linear_dims[index+1])
            self.linear_layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        x = x.mT
        
        if len(x.shape) == 4: #lstm only works on batches of 2D tensors, so need to reshape
            x = x.view(x.shape[0], x.shape[2], x.shape[3])
        
        assert(len(x.shape) == 3)
        
        x, (h_n, c_n) = self.lstm(x)
        h_n = h_n[-1,:,:].unsqueeze(0)
        
        for index, layer in enumerate(self.linear_layers):
            h_n = layer(h_n)
            if index < self.num_linear - 1:
                h_n = self.h_activ(h_n)
        h_n = self.out_activ(h_n)
        return h_n.squeeze(0)
