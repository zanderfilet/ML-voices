import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# SUPER SIMPLE
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        # Encoding
        _, (hidden, _) = self.encoder(x)

        # Repeat the last hidden state to match the input sequence length
        hidden_repeated = hidden.repeat(1, x.size(1), 1)

        # Decoding
        output, _ = self.decoder(hidden_repeated)
        return output.reshape(x.shape)

    
# For Forecasting
class AutoregressiveLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(AutoregressiveLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)

        # Fully connected layer
        output = self.fc(lstm_out[:, -1, :])
        return output

# For deeper forecasting
class AutoregressiveTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, nhead=1):
        super(AutoregressiveTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = TransformerEncoderLayer(hidden_dim, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # Transformer encoder layer
        transformer_out = self.transformer_encoder(embedded)

        # Fully connected layer
        output = self.fc(transformer_out[:, -1, :])
        return output


# Deeper Autoencoder
class Encoder1(nn.Module):
    def __init__(self, seq_len, n_features, n_layers, embedding_dim):
        super(Encoder1, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = 2 * embedding_dim
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        
        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=n_layers,
          batch_first=True 
        )
        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=n_layers,
          batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.seq_len, self.n_features)) 
        x, (_, _) = self.rnn1(x) 
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n[0]

class Decoder1(nn.Module):
    def __init__(self, seq_len, n_features, n_layers, embedding_dim):
        super(Decoder1, self).__init__()

        self.seq_len = seq_len
        self.hidden_dim = 2 * embedding_dim
        self.n_features = n_features
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        
        self.rnn1 = nn.LSTM(
        input_size=embedding_dim,
        hidden_size=embedding_dim,
        num_layers=n_layers,
        batch_first=True
        )
        self.rnn2 = nn.LSTM(
        input_size=embedding_dim,
        hidden_size=self.hidden_dim,
        num_layers=n_layers,
        batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):

        batch_size = x.shape[0]
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_n, cell_n) = self.rnn1(x) 
        x, (hidden_n, cell_n) = self.rnn2(x) 
        x = self.output_layer(x)
        return x

class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, n_layers, embedding_dim):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder1(seq_len, n_features, n_layers, embedding_dim)
        self.decoder = Decoder1(seq_len, n_features, n_layers, embedding_dim)
        
    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)
        return x
    
    
# Less deep autoencoder
# credit: https://github.com/hellojinwoo/TorchCoder/blob/master/autoencoders/rae.py
class Encoder(nn.Module):
    def __init__(self, seq_len, no_features, embedding_size):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features    # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size   # the number of features in the embedded points of the inputs' number of features
        self.LSTM1 = nn.LSTM(
            input_size = no_features,
            hidden_size = self.embedding_size,
            num_layers = 1,
            batch_first=True
        )
        
    def forward(self, x):
        # Outputs: output, (h_n, c_n)
        _, (hidden_state, cell_state) = self.LSTM1(x)  
        last_lstm_layer_hidden_state = hidden_state[-1,:,:]
        return last_lstm_layer_hidden_state
    
    
# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_size, output_size):
        super().__init__()

        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.LSTM1 = nn.LSTM(
            input_size = self.embedding_size,
            hidden_size = 2 * self.embedding_size,
            num_layers = 1,
            batch_first = True
        )
        
        self.fc = nn.Linear(2 * self.embedding_size, self.output_size)
        
    def forward(self, x):
        # input is a embedding_dim x 1 vector
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden_state, cell_state) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, 2 * self.embedding_size))
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out
    
    
# (3) Autoencoder : putting the encoder and decoder together
class LSTM_AE(nn.Module):
    def __init__(self, seq_len, no_features, embedding_dim):
        super().__init__()
        
        self.seq_len = seq_len
        self.no_features = no_features
        self.embedding_dim = embedding_dim

        self.encoder = Encoder(self.seq_len, self.no_features, self.embedding_dim)
        self.decoder = Decoder(self.seq_len, self.embedding_dim, self.no_features)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class TimeStepTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, num_classes):
        super(TimeStepTransformer, self).__init__()
        self.embedding = nn.Linear(16, d_model)
        self.positional_encoding = self._generate_positional_encoding(d_model, 375)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.transformer.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x

    def _generate_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

# Model parameters
d_model = 512
nhead = 8
num_layers = 6
num_classes = 7

# Instantiate the model
model = TimeStepTransformer(d_model, nhead, num_layers, num_classes)

inp = torch.rand(375, 16)

out = model(inp)
print(out.shape)