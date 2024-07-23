

from transformers import GPT2Tokenizer, AutoConfig, OPTForCausalLM

# 1. 8bit加载模型
model_id = "/workspace/model/opt-6.7b"
model = OPTForCausalLM.from_pretrained(model_id, load_in_4bit=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_id)

# 2. 使用 peft 库预定义的工具函数 prepare_model_for_int8_training
# 完成预处理
'''
    (1)将所有非 int8 模块转换为全精度（fp32）以保证稳定性
    (2)为输入嵌入层添加一个 forward_hook，以启用输入隐藏状态的梯度计算
    (3)启用梯度检查点以实现更高效的内存训练
'''
from peft import prepare_model_for_int8_training

model = prepare_model_for_int8_training(model)

memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 3)  # 转换为 GB
print(f"{memory_footprint_mib:.2f}GB")




# 3. LoRA Adapter 配置

# 从peft库导入LoraConfig和get_peft_model函数
from peft import LoraConfig, get_peft_model

# 创建一个LoraConfig对象，用于设置LoRA（Low-Rank Adaptation）的配置参数
config = LoraConfig(
    r=8,  # LoRA的秩，影响LoRA矩阵的大小
    lora_alpha=32,  # LoRA适应的比例因子
    # 指定将LoRA应用到的模型模块，通常是attention和全连接层的投影
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"],
    lora_dropout=0.05,  # 在LoRA模块中使用的dropout率
    bias="none",  # 设置bias的使用方式，这里没有使用bias
    task_type="CAUSAL_LM"  # 任务类型，这里设置为因果(自回归）语言模型
)
# 使用get_peft_model函数和给定的配置来获取一个PEFT模型
model = get_peft_model(model, config)
# 打印出模型中可训练的参数
model.print_trainable_parameters()

# 4. 数据处理
from datasets import load_dataset
dataset = load_dataset("/workspace/data/english_quotes")
tokenized_dataset = dataset.map(lambda samples: tokenizer(samples["quote"]), batched=True)

from transformers import DataCollatorForLanguageModeling

# 数据收集器，用于处理语言模型的数据，这里设置为不使用掩码语言模型(MLM)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# 5. 微调模型
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
        output_dir=f"/{model_id}-lora",  # 指定模型输出和保存的目录
        per_device_train_batch_size=16,  # 每个设备上的训练批量大小
        learning_rate=2e-4,  # 学习率
        fp16=True,  # 启用混合精度训练，可以提高训练速度，同时减少内存使用
        logging_steps=20,  # 指定日志记录的步长，用于跟踪训练进度
        max_steps=100, # 最大训练步长
        # num_train_epochs=1  # 训练的总轮数
    )
trainer = Trainer(
    model=model,  # 指定训练时使用的模型
    train_dataset=tokenized_dataset["train"],  # 指定训练数据集
    args=training_args,
    data_collator=data_collator,
)


model.use_cache = False
trainer.train()


# 6. 保存模型
model_path = f"{model_id}-lora-int8"
#trainer.save_model(model_path)
model.save_pretrained(model_path)

# 7. 使用LoRA模型
lora_model = trainer.model
text = "Two things are infinite: "
inputs = tokenizer(text, return_tensors="pt").to(0)

out = lora_model.generate(**inputs, max_new_tokens=48)
print(tokenizer.decode(out[0], skip_special_tokens=True))
