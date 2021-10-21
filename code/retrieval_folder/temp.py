from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import wandb
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from bm25_first_passage_from_wiki import get_bm25_passage


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


set_seed(42)  # magic number :)

try:
    dataset = load_from_disk("mrc-level2-nlp-04/data/train_dataset")
except:
    dataset = load_from_disk("../../data/train_dataset")

model_checkpoint = "klue/roberta-small"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
configuration = AutoConfig.from_pretrained(model_checkpoint)
emb_size = configuration.max_position_embeddings
training_dataset = dataset['train'][:1]






answers = training_dataset['answers']
ans_seqs = []
for answer in answers:
    ans_seqs.append(answer["text"][0])

bm25_passages = get_bm25_passage(training_dataset["question"], ans_seqs)

print(f"질문: {training_dataset['question']},\n\n정답이 포함된 문서: {training_dataset['context']},\n\n정답: {training_dataset['answers']},\n\nBM25: {bm25_passages}")