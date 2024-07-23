from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_path = "/workspace/model/opt-2.7b"

# 量化配置
quantization_config = GPTQConfig(
     bits=4, # 量化精度
     group_size=128,
     dataset="wikitext2", # 量化器支持的默认数据集（包括['wikitext2','c4','c4-new','ptb','ptb-new']）
     # dataset 也可以为自己的数据集
     desc_act=False,
)
# 量化模型
quant_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map='auto')
# 检查量化模型正确性
print(quant_model.model.decoder.layers[0].self_attn.q_proj.__dict__)

# 保存模型权重
quant_path = "/workspace/model/opt-2.7b-gptq"
quant_model.save_pretrained(quant_path)

# 文本生成
tokenizer = AutoTokenizer.from_pretrained(model_path)
text = "Merry Christmas! I'm glad to"
inputs = tokenizer(text, return_tensors="pt").to(0)
out = quant_model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))