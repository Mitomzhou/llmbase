import torch.nn as nn
import torchinfo

class RNNModel(nn.Module):
    """ encoder - lstm - decoder """

    def __init__(self, rnn_type, vocab_size, emsize, nhidden, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = nn.Embedding(vocab_size, nhidden)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emsize, nhidden, nlayers, dropout=dropout)  # getattr内置函数，获取nn的rnn_type的值如果没有，触发异常
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(emsize, nhidden, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.decoder = nn.Linear(nhidden, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.rnn_type = rnn_type
        self.nhidden = nhidden
        self.nlayers = nlayers

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, batch_size, self.nhidden), weight.new_zeros(self.nlayers, batch_size, self.nhidden))
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhidden)

    def forward(self, input, hidden):
        embedding = self.drop(self.encoder(input))
        output, hidden = self.rnn(embedding, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        return decoded, hidden



# model = RNNModel("LSTM", 1000, 200, 300, 3)
# torchinfo.summary(model=model)