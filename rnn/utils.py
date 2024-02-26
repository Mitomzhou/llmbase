import torch

def batchify(data, batch_size, device):
    """
    数据批量化处理
    若 batch_size = 4,  [a b c d e f g h i j k .... s t u v w x y z] ->
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    """
    nbacth = data.size(0) // batch_size
    data = data.narrow(0, 0, nbacth * batch_size) # 去掉尾部几个单词
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)   # 减1主要是最后一批量data需要留一行给target
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach() # 阻断反向梯度传播
    else:
        return tuple(repackage_hidden(v) for v in h)