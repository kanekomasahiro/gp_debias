import torch
import torch.nn as nn
from torch import cuda

torch.manual_seed(0)
cuda.manual_seed_all(0)

class Encoder(nn.Module):

    def __init__(self, emb_size, hidden_size, dropout_rate):
        super(Encoder, self).__init__()
        self.hidden_layer = nn.Linear(emb_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.tanh = nn.Tanh()

    def forward(self, emb):
        hidden = self.tanh(self.hidden_layer(self.dropout(emb)))

        return hidden


class Classifier(nn.Module):

    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden):
        output = self.output_layer(hidden)
        pre = self.sigmoid(output)

        return pre


class Decoder(nn.Module):

    def __init__(self, hidden_size, emb_size, dropout_rate):
        super(Decoder, self).__init__()
        self.output_layer = nn.Linear(hidden_size, emb_size)
        self.tanh = nn.Tanh()

    def forward(self, hidden):
        pre = self.tanh(self.output_layer(hidden))

        return pre
