from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import evaluate
import numpy as np


yelp_review_full = "/workspace/data/yelp_review_full"
model_checkpoint = "/workspace/model/bert-base-cased"

# 1.加载数据集
dataset = load_dataset(yelp_review_full)
print(dataset)

# 2.加载模型
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 3.对原始数据进行预处理
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=32).select(range(1000))
test_dataset = tokenized_datasets["test"].shuffle(seed=32).select(range(1000))

# 4.带有序列分类的模型加载
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=5)

model_dir = "/workspace/model/bert-base-cased-finetune-yelp"

# 5.定义metric函数定义准确度
metric = evaluate.load("../third/accuracy.py")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 6.定义训练参数
training_args = TrainingArguments(output_dir=model_dir,
                                  per_device_train_batch_size=16,
                                  num_train_epochs=3,
                                  logging_steps=100,
                                  evaluation_strategy="steps",  # 在每个 steps 后进行评估
                                  save_steps=500,
                                  eval_steps=500,
                                  )
# 7.微调bert模型
trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=test_dataset,
                  compute_metrics=compute_metrics)

trainer.train()
trainer.save_model()

# 8. 输出评估结果
results = trainer.evaluate(model_dir)
print("Results:", results)