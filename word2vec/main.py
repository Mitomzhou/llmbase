import argparse
import yaml
import os
import torch
import torch.nn as nn

from utils import (get_dataloader_and_vocab,
                   get_model_class,
                   get_optimizer_class,
                   get_lr_scheduler,
                   save_vocab)
from trainer import Trainer


def train(config):
    os.makedirs(config["model_dir"])

    train_dataloader, vocab = get_dataloader_and_vocab("cbow","WikiText2", "train", "data/", 96, True, None)
    val_dataloader, vocab = get_dataloader_and_vocab("cbow", "WikiText2", "valid", "data/", 96, True, vocab)

    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    model_class = get_model_class(config["model_name"])
    model = model_class(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)