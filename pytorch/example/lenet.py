import torch
from torch import nn , optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# 定义超参数
learning_rate = 1e-3
batch_size    = 256
epoches_num   = 20

# 加载cpu/gpu设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is enable!")

# 加载数据集
train_dataset = datasets.MNIST( root='../data', train=True, transform=transforms.ToTensor(), download=True)
train_loader  = DataLoader( train_dataset, batch_size=batch_size, shuffle=True )
test_dataset  = datasets.MNIST( root='../data', train=False, transform=transforms.ToTensor())
test_loader   = DataLoader(test_dataset , batch_size=batch_size, shuffle=False)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
        self.fullcon1 = nn.Linear(16*5*5, 120)
        self.fullcon2 = nn.Linear(120, 84)
        self.fullcon3 = nn.Linear(84, 10)
        #self.dropout = nn.Dropout(0.2)
        #self.dropout = nn.Dropout(0.2)

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = X.view(X.size(0), -1)
        X = self.fullcon1(X)
        X = F.relu(X)
        X = self.fullcon2(X)
        X = F.relu(X)
        out = self.fullcon3(X)
        return out


model = LeNet().to(device)
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
        test_loss /= len(test_loader.dataset)    #损失和/加载的数据集的总数
        Accuracy = correct / len(text_loader.dataset)
        print("Val Acc: {:.3f} | Val Loss: {:6f}\n".format(Accuracy, test_loss))

import time
start=time.time()

# 开始训练和验证
for epoch in range(1, epoches_num+1):
    train_model(model, device, train_loader, optimizer, epoch)
    eval_model(model, device, test_loader)

end=time.time()
print('程序运行时间为: {:.3f} Seconds'.format(end-start))
# 保存模型
torch.save(model, 'lenet.pt')