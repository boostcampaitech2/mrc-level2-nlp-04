import nltk
import random
import torch
import numpy as np

from datasets import load_dataset, load_metric
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from utils_qa import check_no_error, get_args, set_seed_everything, get_models, get_data

# datasets = load_dataset()
# metric = load_metric()

def preprocess_function(examples):
    inputs = [f"question: {q}  context: {c} </s>" for q, c in zip(examples["question"], examples["context"])]
    targets = [f'{a["text"][0]} </s>' for a in examples['answers']]
    model_inputs = tokenizer(
        inputs,
        max_length=max_source_length,
        padding=padding,
        truncation=True
    )

    # targets(label)을 위해 tokenizer 설정
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=max_target_length,
            padding=padding,
            truncation=True
        )

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["example_id"] = []
    for i in range(len(model_inputs["labels"])):
        model_inputs["example_id"].append(examples["id"][i])
    return model_inputs

def postprocess_text(preds, labels):
    """
    postprocess는 nltk를 이용합니다.
    Huggingface의 TemplateProcessing을 사용하여
    정규표현식 기반으로 postprocess를 진행할 수 있지만
    해당 미션에서는 nltk를 이용하여 간단한 후처리를 진행합니다
    """
  
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
      
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
  
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
  
    # 간단한 post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
  
    formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"].select(range(max_val_samples)))]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"].select(range(max_val_samples))]
  
    result = metric.compute(predictions=formatted_predictions, references=references)
    return result

class GenerationBasedModel():
    def __init__(self, ):




model_name = "google/mt5-small"

config = AutoConfig.from_pretrained(
    model_name,
    cache_dir=None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=None,
    use_fast=True,
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    config=config,
    cache_dir=None,
)

max_source_length = 1024
