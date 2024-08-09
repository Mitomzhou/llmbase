from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor
from transformers import AutoModelForSpeechSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from peft import prepare_model_for_int8_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

##################################################################
# LoRA微调whisper-large-v3:int8, 数据为common_voice_11_0
##################################################################

model_dir = "/workspace/model/whisper-large-v3"
model_dir_lora_int8 = "/workspace/model/whisper-large-v3-int8"
language = "Chinese (China)"
language_abbr = "zh-CN"
task = "transcribe"
dataset_name = "mozilla-foundation/common_voice_11_0"
batch_size = 64

##################################################################
# 1.下载数据集
##################################################################
common_voice = DatasetDict()
common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train", trust_remote_code=True)
common_voice["validation"] = load_dataset(dataset_name, language_abbr, split="validation", trust_remote_code=True)
print(common_voice)

##################################################################
# 2.预处理训练数据集
##################################################################
# 从预训练模型中加载特征提取器，分词器，处理器
feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir, language=language, task=task)
processor = AutoProcessor.from_pretrained(model_dir, language=language, task=task)
# 移除数据集中不必要的字段
common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
)
print(common_voice["train"][0])

# common_voice 48k， whisper训练数据为16k，需要降频
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))


# 数据预处理函数
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

# 数据shuffle
common_voice["train"] = common_voice["train"].shuffle(seed=16).select(range(640))
common_voice["validation"] = common_voice["validation"].shuffle(seed=16).select(range(320))

# 数据map
tokenized_common_voice = common_voice.map(prepare_dataset, num_proc=8)


# 定义一个针对语音到文本任务的数据整理器类
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any  # 处理器结合了特征提取器和分词器

    # 整理器函数，将特征列表处理成一个批次
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 从特征列表中提取输入特征，并填充以使它们具有相同的形状
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # 从特征列表中提取标签特征（文本令牌），并进行填充
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 使用-100替换标签中的填充区域，-100通常用于在损失计算中忽略填充令牌
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # 如果批次中的所有序列都以句子开始令牌开头，则移除它
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # 将处理过的标签添加到批次中
        batch["labels"] = labels

        return batch  # 返回最终的批次，准备好进行训练或评估


# 用给定的处理器实例化数据整理器
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

##################################################################
# 3.模型准备
##################################################################
# int8精度加载预训练模型
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, load_in_8bit=True, device_map="auto")
# 设置模型配置中的forced_decoder_ids属性为None
model.config.forced_decoder_ids = None  # 这通常用于指定在解码（生成文本）过程中必须使用的特定token的ID，设置为None表示没有这样的强制要求
# 设置模型配置中的suppress_tokens列表为空
model.config.suppress_tokens = []  # 这用于指定在生成过程中应被抑制（不生成）的token的列表，设置为空列表表示没有要抑制的token

"""
    将所有非 int8 精度模块转换为全精度（fp32）以保证稳定性
    为输入嵌入层添加一个 forward_hook，以启用输入隐藏状态的梯度计算
    启用梯度检查点以实现更高效的内存训练
"""
model = prepare_model_for_int8_training(model)


# 创建一个LoraConfig对象，用于设置LoRA（Low-Rank Adaptation）的配置参数
config = LoraConfig(
    r=4,  # LoRA的秩，影响LoRA矩阵的大小
    lora_alpha=64,  # LoRA适应的比例因子
    # 指定将LoRA应用到的模型模块，通常是attention和全连接层的投影。
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,  # 在LoRA模块中使用的dropout率
    bias="none",  # 设置bias的使用方式，这里没有使用bias
)

peft_model = get_peft_model(model, config)
# 打印 LoRA 微调训练的模型参数
peft_model.print_trainable_parameters()

##################################################################
# 4.模型训练
##################################################################
training_args = Seq2SeqTrainingArguments(
    output_dir=model_dir_lora_int8,
    per_device_train_batch_size=batch_size,
    learning_rate=1e-3,
    #num_train_epochs=3,
    #evaluation_strategy="epoch",
    # warmup_steps=50,  # 在训练初期增加学习率的步数，有助于稳定训练
    # fp16=True,  # 启用混合精度训练，可以提高训练速度，同时减少内存使用
    per_device_eval_batch_size=batch_size,
    generation_max_length=128, # 生成任务的最大长度
    logging_steps = 100,  # 日志记录步数
    remove_unused_columns=False,
    label_names=["labels"],
    # evaluation_strategy="steps",
    # eval_steps=25,

    max_steps=5,# 训练总步数
    evaluation_strategy="steps",
    eval_steps=5, # 评估步数
)

# 实例化训练器
trainer = Seq2SeqTrainer(
    args=training_args,
    model=peft_model,
    train_dataset=tokenized_common_voice["train"],
    eval_dataset=tokenized_common_voice["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)
peft_model.config.use_cache = False

# 训练
trainer.train()

##################################################################
# 5.保存lora模型
##################################################################
trainer.save_model(model_dir_lora_int8)
peft_model.eval()




