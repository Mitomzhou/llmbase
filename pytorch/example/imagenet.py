import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import os

import net.AlexNet as AlexNet
import net.VGGNet as VGGNet
import net.ResNet as ResNet

# 定义超参数
learning_rate = 1e-3
batch_size    = 64
epoches_num   = 10

# 加载cpu/gpu设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is enable!")

def load_ImageNet(ImageNet_PATH, batch_size=64, workers=8, pin_memory=True):
    traindir = os.path.join(ImageNet_PATH, 'trainlite')
    valdir = os.path.join(ImageNet_PATH, 'vallite')
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
# model = AlexNet.alexnet.to(device)
model = VGGNet.vggnet.to(device)
# model = ResNet.resnet.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    train_acc = 0.0
    for idx,(data ,target) in enumerate(train_loader):
        data, target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # 计算训练精度
        _, pred = output.max(1)
        num_correct = pred.eq(target).sum()
        train_acc += num_correct.item()

        if idx % 50 == 0:
            print("Epoch: {:0>4d}/{:0>4d} | Batch {:0>4d}/{:0>4d} | Loss: {:.6f}".format(epoch, epoches_num, idx, int(len(train_dataset)/batch_size), loss.item()))
    print("Train Acc: {:.3f} | ".format(train_acc / len(train_dataset)), end="")

def eval_model(model, device, text_loader):
    model.eval()
    correct = 0.0
    global Accuracy
    test_loss = 0.0
    with torch.no_grad(): # 不会计算梯度，也不会进行反向传播
        for data, target in text_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output,target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item() #累计正确的值
        test_loss /= len(val_loader.dataset)    #损失和/加载的数据集的总数
        Accuracy = correct / len(text_loader.dataset)
        print("Val Acc: {:.3f} | Val Loss: {:6f}\n".format(Accuracy, test_loss))

import time
start=time.time()

# 开始训练和验证
for epoch in range(1, epoches_num+1):
    train_model(model, device, train_loader, optimizer, epoch)
    eval_model(model, device, val_loader)

end=time.time()
print('程序运行时间为: {:.3f} Seconds'.format(end-start))
# 保存模型
torch.save(model, 'imagenet.pt')