from transformers import pipeline

pipe = pipeline(task="sentiment-analysis", model="/home/skyrover/LLM/model/distilbert-base-uncased-finetuned-sst-2-english")
text = "weather is good today"
prediction = pipe(text)
print(prediction)

text_list = [
    "Today Shanghai is really cold.",
    "I think the taste of the garlic mashed pork in this store is average.",
    "You learn things really quickly. You understand the theory class as soon as it is taught."
]
prediction = pipe(text_list)
print(prediction)


# 使用 `model` 参数指定模型
transcriber = pipeline(task="automatic-speech-recognition", model="/home/skyrover/LLM/model/whisper-smalll")