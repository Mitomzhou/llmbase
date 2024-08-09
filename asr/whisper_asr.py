from transformers import pipeline
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoProcessor
from transformers import AutoModelForSpeechSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutomaticSpeechRecognitionPipeline
from datasets import load_dataset, DatasetDict, Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from peft import prepare_model_for_int8_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel


model_dir = "/workspace/model/whisper-large-v3"
model_dir_lora_int8 = "/workspace/model/whisper-large-v3-int8"
language = "Chinese (China)"
language_abbr = "zh-CN"
language_decode = "chinese"
task = "transcribe"

##################################################################
# 1.原始模型语音识别
##################################################################
def whisper_asr():
    transcriber = pipeline(task="automatic-speech-recognition", model=model_dir)
    text = transcriber("../data/test_zh.flac")
    text2= transcriber("../data/common_voice_en_436.mp3")
    text3= transcriber("../data/common_voice_ja_19779220.mp3")
    print(text)
    print(text2)
    print(text3)

##################################################################
# 2.LoRA微调后模型语音识别
##################################################################
def whisper_lora_int8_asr():
    # 使用PeftModel加载LoRA微调后的Whisper模型
    peft_config = PeftConfig.from_pretrained(model_dir_lora_int8)
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        peft_config.base_model_name_or_path,
        load_in_8bit=True,
        device_map="auto"
    )
    peft_model =PeftModel.from_pretrained(base_model, model_dir_lora_int8)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
    processor = AutoProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
    feature_extractor = processor.feature_extractor

    pipeline = AutomaticSpeechRecognitionPipeline(model=peft_model, tokenizer=tokenizer, feature_extractor=feature_extractor)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language_decode, task=task)
    test_audio = "../data/test_zh.flac"
    with torch.cuda.amp.autocast():
        # 必须加这个不然报错 RuntimeError: Expected is_sm90 || is_sm8x || is_sm75 to be true, but got false.
        torch.backends.cuda.enable_flash_sdp(False)
        text = pipeline(test_audio, max_new_tokens=255)["text"]
        print(text)

whisper_lora_int8_asr()





