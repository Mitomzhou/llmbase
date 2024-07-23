from transformers import pipeline, AutoTokenizer, AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM
from transformers import AwqConfig, AutoConfig

# 测试文本生成任务
model_path = "/workspace/model/opt-125m"
generator = pipeline('text-generation', model=model_path, device=0, do_sample=True, num_return_sequences=3)
print(generator("The woman worked as a"))

# 使用 AutoAWQ 量化模型
quant_path = "/workspace/model/opt-125m-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit":4, "version":"GEMM"}
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.quantize(tokenizer, quant_config=quant_config)

# Transformer兼容性配置
quantization_config = AwqConfig(
    bits=quant_config["w_bit"],
    group_size=quant_config["q_group_size"],
    zero_point=quant_config["zero_point"],
    version=quant_config["version"].lower(),
).to_dict()

# 模型量化
model.model.config.quantization_config = quantization_config

# 保存权重和分词器
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

# 加载量化后的模型
tokenizer = AutoTokenizer.from_pretrained(quant_path)
model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="cuda").to(0)

def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to(0)
    out = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# 测试输出
result = generate_text("Merry Christmas! I'm glad to")
print(result)
result = generate_text("The woman worked as a")
print(result)