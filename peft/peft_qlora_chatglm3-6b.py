from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import torch
from typing import List, Dict, Optional

'''
    数据准备
        下载数据集
        设计 Tokenizer 函数处理样本（map、shuffle、flatten）
        自定义批量数据处理类 DataCollatorForChatGLM
    训练模型
        加载 ChatGLM3-6B 量化模型
        PEFT 量化模型预处理（prepare_model_for_kbit_training）
        QLoRA 适配器配置（TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING）
        微调训练超参数配置（TrainingArguments）
        开启训练（trainer.train)
        保存QLoRA模型（trainer.model.save_pretrained)
    模型推理
        加载 ChatGLM3-6B 基础模型
        加载 ChatGLM3-6B QLoRA 模型（PEFT Adapter）
        微调前后对比
'''

# 0. 定义全局变量
model_id = "/workspace/model/chatglm3-6b"
train_data_path = "/workspace/data/adgen"
eval_data_path = None
seed = 8
max_input_length = 512  # 最大输入token数
max_output_length = 1536   # 最大输出token数
lora_rank = 4  # lora的秩
lora_alpha = 32
lora_dropout = 0.05
resume_from_checkpoint = None
prompt_text = ''    # 所有数据前的指令文本
compute_dtype = 'fp32'    # 计算数据类型


# 1. 数据准备
dataset = load_dataset(train_data_path)
print(dataset['train'][0])
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

def tokenize_func(example, tokenizer, ignore_label_id=-100):
    question = prompt_text + example['content']

    # 检测提示词输入
    if example.get('input', None) and example['input'].strip():
        question += f'\n{example["input"]}'

    answer = example['summary']

    # 对问题和答案文本进行tokenize处理
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    # 如果tokenize后的长度超过最大长度限制，则进行截断
    if len(q_ids) > max_input_length - 2:  # 保留空间给gmask和bos标记
        q_ids = q_ids[:max_input_length - 2]
    if len(a_ids) > max_output_length - 1:  # 保留空间给eos标记
        a_ids = a_ids[:max_output_length - 1]

    # 构建模型的输入格式
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    question_length = len(q_ids) + 2  # 加上gmask和bos标记

    # 构建标签，对于问题部分的输入使用ignore_label_id进行填充
    labels = [ignore_label_id] * question_length + input_ids[question_length:]

    # input_ids :  问题+回答
    # labels    :  问题部分用-100+回答
    # input_ids 和 labels一样长
    return {'input_ids': input_ids, 'labels': labels}


column_names = dataset['train'].column_names
tokenized_dataset = dataset['train'].map(
    lambda example: tokenize_func(example, tokenizer),
    batched=False,
    remove_columns=column_names
)

# 数据集shuffle flatten
tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
tokenized_dataset = tokenized_dataset.flatten_indices()

# 批处理数据
class DataCollatorForChatGLM:

    """
        pad_token_id (int): 用于填充(padding)的token ID。
        max_length (int): 单个批量数据的最大长度限制。
        ignore_label_id (int): 在标签中用于填充的ID，默认为-100。
    """
    def __init__(self, pad_token_id:int, max_length: int = 2048, ignore_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_label_id = ignore_label_id
        self.max_length = max_length

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """
        处理批量数据。

        参数:
        batch_data (List[Dict[str, List]]): 包含多个样本的字典列表。

        返回:
        Dict[str, torch.Tensor]: 包含处理后的批量数据的字典。
        """
        # 计算批量中每个样本的长度
        len_list = [len(d['input_ids']) for d in batch_data]
        batch_max_len = max(len_list)  # 找到最长的样本长度

        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d  # 计算需要填充的长度
            # 添加填充，并确保数据长度不超过最大长度限制
            ids = d['input_ids'] + [self.pad_token_id] * pad_len
            label = d['labels'] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[:self.max_length]
                label = label[:self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))

        # 将处理后的数据堆叠成一个tensor
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return {'input_ids': input_ids, 'labels': labels}

# 准备数据整理器
data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)


# 2. 训练数据

# 使用 nf4 量化数据类型加载模型，开启双量化配置，以bf16混合精度训练，预估显存占用接近4GB
_compute_dtype_map = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16
}

# QLoRA 量化配置
q_config = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=_compute_dtype_map['bf16'])

model = AutoModel.from_pretrained(model_id,
                                  quantization_config=q_config,
                                  device_map='auto',
                                  trust_remote_code=True)

# 获取当前模型占用的 GPU显存
memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB
print(f"{memory_footprint_mib:.2f}MiB")

from peft import TaskType, LoraConfig, get_peft_model, prepare_model_for_kbit_training
kbit_model = prepare_model_for_kbit_training(model)

from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['chatglm']

# LoRA 适配器配置
lora_config = LoraConfig(
    target_modules=target_modules,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias='none',
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM
)
qlora_model = get_peft_model(kbit_model, lora_config)
qlora_model.print_trainable_parameters()


# from transformers import TrainingArguments, Trainer
#
# training_args = TrainingArguments(
#     output_dir=f"/workspace/model/{model_id}-qlora",          # 输出目录
#     per_device_train_batch_size=32,                     # 每个设备的训练批量大小
#     gradient_accumulation_steps=8,                     # 梯度累积步数
#     # per_device_eval_batch_size=8,                      # 每个设备的评估批量大小
#     learning_rate=1e-3,                                # 学习率
#     num_train_epochs=1,                                # 训练轮数
#     lr_scheduler_type="linear",                        # 学习率调度器类型
#     warmup_ratio=0.1,                                  # 预热比例
#     logging_steps=10,                                 # 日志记录步数
#     save_strategy="steps",                             # 模型保存策略
#     save_steps=100,                                    # 模型保存步数
#     # evaluation_strategy="steps",                       # 评估策略
#     # eval_steps=500,                                    # 评估步数
#     optim="adamw_torch",                               # 优化器类型
#     fp16=True,                                        # 是否使用混合精度训练
# )
#
# trainer = Trainer(
#         model=qlora_model,
#         args=training_args,
#         train_dataset=tokenized_dataset,
#         data_collator=data_collator
#     )


from transformers import TrainingArguments, Trainer

training_demo_args = TrainingArguments(
    output_dir=f"{model_id}-qlora2",          # 输出目录
    per_device_train_batch_size=16,                     # 每个设备的训练批量大小
    gradient_accumulation_steps=4,                     # 梯度累积步数
    learning_rate=1e-3,                                # 学习率
    max_steps=100,                                     # 训练步数
    lr_scheduler_type="linear",                        # 学习率调度器类型
    warmup_ratio=0.1,                                  # 预热比例
    logging_steps=10,                                 # 日志记录步数
    save_strategy="steps",                             # 模型保存策略
    save_steps=20,                                    # 模型保存步数
    optim="adamw_torch",                               # 优化器类型
    fp16=True,                                        # 是否使用混合精度训练
)

trainer = Trainer(
        model=qlora_model,
        args=training_demo_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )




trainer.train()
trainer.model.save_pretrained(f"{model_id}-qlora2")