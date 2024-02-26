import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import os

def load_ImageNet(ImageNet_PATH, batch_size=16, workers=3, pin_memory=True):
    traindir = os.path.join(ImageNet_PATH, 'train')
    valdir = os.path.join(ImageNet_PATH, 'val')
    print('traindir = ', traindir)
    print('valdir = ', valdir)

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalizer
        ])
    )

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer
        ])
    )
    print('train_dataset = ', len(train_dataset))
    print('val_dataset   = ', len(val_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, train_dataset, val_dataset

train_loader, val_loader, train_dataset, val_dataset = load_ImageNet('/data/imagenet')
classes = train_dataset.classes
print("Classes:", classes)
print(train_dataset.class_to_idx)
# print(train_dataset.imgs)