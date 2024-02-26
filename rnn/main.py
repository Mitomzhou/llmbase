import argparse
import yaml
import math
import time
import torch
import torch.nn as nn
from torch import optim

import data
import model
from utils import  (batchify, get_batch, repackage_hidden)


##############################################################
# 1.读取配置文件
##############################################################
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU')
parser.add_argument('--config', type=str, required=True, help='path to yaml config')
args = parser.parse_args()
with open(args.config, 'r') as stream:
    config = yaml.safe_load(stream)

torch.manual_seed(config['seed']) # 设置随机种子，以便每次实验得到固定随机数便于验证

if torch.cuda.is_available():
    if not config['cuda']:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda.')
device = torch.device('cuda' if config['cuda'] else 'cpu')

##############################################################
# 2.加载语料库数据
##############################################################
corpus = data.Corpus(config['data'])
eval_batch_size = 10
train_data = batchify(corpus.train, config['batch_size'], device)
val_data = batchify(corpus.valid, eval_batch_size, device)
test_data = batchify(corpus.test, eval_batch_size, device)

##############################################################
# 3.初始化模型
##############################################################
vocab_size = len(corpus.dictionary)
# if config['model'] == 'LSTM':
#     if config['model'] == 'LSTM':
model = model.RNNModel(config['model'], vocab_size, config['emsize'], config['nhidden'], config['nlayers'], config['dropout']).to(device)

criterion = nn.CrossEntropyLoss()

##############################################################
# 4.定义训练和评估函数
##############################################################
def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(config['batch_size'])
    for batch, i in enumerate(range(0, train_data.size(0) - 1, config['seq_len'])):  # 由于我们已经把train_data整理成batch构成的数组了，这里直接进行迭代即可。
        data, targets = get_batch(train_data, i, config['seq_len'])
        model.zero_grad()
        hidden = repackage_hidden(hidden) # 切断其中重叠的反向传播, 否则bachword()报错
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip']) # 梯度裁剪防止梯度爆炸
        # 手动更新参数, 可以用optimizer改,目前没有搞出来
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % config['log_interval'] == 0 and batch > 0:
            cur_loss = total_loss / config['log_interval']
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // config['seq_len'], lr,elapsed * 1000 / config['log_interval'], cur_loss, math.exp(cur_loss)))  # math.exp(cur_loss):语言模型困惑度
            total_loss = 0
            start_time = time.time()


def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, config['seq_len']):
            data, targets = get_batch(data_source, i, config['seq_len'])
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

##############################################################
# 5.开始训练
##############################################################
best_val_loss = None
lr = config['lr']
try:
    for epoch in range (1, config['epochs']+1):
        epoch_start_time = time.time()
        train() # 训练
        val_loss = evaluate(val_data) # 验证
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(
            epoch, (time.time() - epoch_start_time),val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(config['save'], 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0
except KeyboardInterrupt:  # ctrl + C退出
    print('-' * 89)
    print('Exiting from training early')

##############################################################
# 6.测试
##############################################################
with open(config['save'], 'rb') as f:
    model = torch.load(f)
    if config['model'] in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters() # 将模型参数扁平化，使其转化为一维形式，能够加快计算速度

test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
print('=' * 89)


##############################################################
# 7.预测句子后一个词: 感觉没有把之前的hidden加载进去
##############################################################
def pred():
    input = 'a military unit and take part in missions against enemy forces'
    idss = []
    ids = []
    words = input.split()

    for word in words:
        ids.append(corpus.dictionary.word2idx[word])
    ids = torch.tensor(ids).type(torch.int64).to(device)
    print(ids)

    model = torch.load('model.pt', map_location=device)
    model.to(device)
    model.eval()
    total_loss = 0.
    hidden = None
    output, hidden = model(ids,hidden)
    print(output)
    print(output.shape)
    output_tensor = torch.nn.functional.softmax(output, dim=1)
    print(output_tensor)
    max_index = torch.argmax(output_tensor, dim=1)
    print(max_index)

    for i in max_index:
        print(corpus.dictionary.idx2word[i.item()])

