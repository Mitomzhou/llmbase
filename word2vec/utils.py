import os

import torch
from functools import partial
from torch.utils.data import DataLoader
import torch.optim as optim

import torchtext
from torchtext.datasets import WikiText2, WikiText103
from torchtext.data.utils import get_tokenizer

import constants
from model import CBOW_Model, SkipGram_Model
def get_english_tokenizer():
    """
    Documentation:
    https://pytorch.org/text/stable/_modules/torchtext/data/utils.html#get_tokenizer
    """
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer

def build_vocab(data_iter, tokenizer):
    vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, data_iter), specials=["<unk>"], min_freq=constants.MIN_WORD_FREQUENCY)
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def get_data_iterator(ds_name, ds_type, data_dir):
    if ds_name == "WikiText2":
        data_iter = WikiText2(root=data_dir, split=(ds_type))  # split:切分的数据集（train,valid,test）
    elif ds_name == "WikiText103":
        data_iter = WikiText103(root=data_dir, split=(ds_type))
    else:
        raise ValueError("Choose dataset from: WikiText2, WikiText103")
    data_iter = torchtext.data.to_map_style_dataset(data_iter)
    return data_iter


def collate_cbow(batch, text_pipeline):
    batch_input, batch_output = [], []
    for text in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < constants.CBOW_N_WORDS * 2 + 1:
            continue
        if constants.MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:constants.MAX_SEQUENCE_LENGTH]

        for idx in range(len(text_tokens_ids) - constants.CBOW_N_WORDS * 2):
            token_id_sequence = text_tokens_ids[idx : (idx + constants.CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(constants.CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def collate_skipgram(batch, text_pipeline):
    pass


def get_dataloader_and_vocab(model_name, ds_name, ds_type, data_dir, batch_size, shuffle, vocab=None):
    data_iter = get_data_iterator(ds_name, ds_type, data_dir)
    tokenizer = get_english_tokenizer()

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer)
    text_pipeline = lambda x : vocab(tokenizer(x))

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    dataloader = DataLoader(data_iter, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=partial(collate_fn, text_pipeline=text_pipeline)) # functools.partial  原函数的部分参数固定了初始值，新的调用只需要传递其它参数
    return dataloader, vocab


def get_model_class(model_name: str):
    if model_name == "cbow":
        return CBOW_Model
    elif model_name == "skipgram":
        return SkipGram_Model
    else:
        raise ValueError("Choose model_name from: cbow, skipgram")
        return

def get_optimizer_class(name: str):
    if name == "Adam":
        return optim.Adam
    else:
        raise ValueError("Choose optimizer from: Adam")
        return

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate,
    so thatlearning rate after the last epoch is 0.
    """
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler


def save_vocab(vocab, model_dir: str):
    """Save vocab file to `model_dir` directory"""
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)
# dataloader, vocab = get_dataloader_and_vocab("cbow","WikiText2", "valid", "data/", 96, True, None)