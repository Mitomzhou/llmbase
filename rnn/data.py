import os
import io
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}  # dict结构
        self.idx2word = []  # str array 结构

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    """ 将语料文件转成词表向量 """
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        # self.lite = self.tokenize(os.path.join(path, 'lite.txt'))

    def tokenize(self, path):
        """Tokenize a text file."""
        assert os.path.exists(path)
        with io.open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        with io.open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)  # 将整个文件所有句子变成一行tensor(1,_),长度为整个文件单词数

        return ids