import torch.nn as nn


class CBOW_Model(nn.Module):

    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300)
        self.linear = nn.Linear(in_features=300, out_features=vocab_size)

    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

class SkipGram_Model(nn.Module):

    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=300)
        self.linear = nn.Linear(in_features=300, out_features=vocab_size)

    def _slow_forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


