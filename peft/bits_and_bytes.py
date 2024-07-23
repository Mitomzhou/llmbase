from transformers import AutoModelForCausalLM

model_id = "/workspace/model/opt-2.7b"

# 1、load_in_8bit或load_in_4bit参数, 便可对模型进行量化
model_4bit = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)

#print(model_4bit)
memory_footprint_bytes = model_4bit.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB
print(f"{memory_footprint_mib:.2f}MiB")


# 2、使用 NF4 精度加载模型
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)
model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
# 获取当前模型占用的 GPU显存
memory_footprint_bytes = model_nf4.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB
print(f"{memory_footprint_mib:.2f}MiB")


# 3、使用双量化加载模型

from transformers import BitsAndBytesConfig
import torch
double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
memory_footprint_bytes = model_double_quant.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB
print(f"{memory_footprint_mib:.2f}MiB")


# 4、使用 QLoRA 所有量化技术加载模型
qlora_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_qlora = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=qlora_config)

# 获取当前模型占用的
memory_footprint_bytes = model_qlora.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB
print(f"{memory_footprint_mib:.2f}MiB")

