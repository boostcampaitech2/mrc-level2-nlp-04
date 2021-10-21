import os
import json
from datasets import load_from_disk
import time
import faiss
import pickle
import numpy as np
import pandas as pd

import hashlib 
import pprint

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from datasets import (
    load_metric,
    load_from_disk,
    Sequence,
    Value,
    Features,
    Dataset,
    DatasetDict,
)

from transformers import(
            BertModel, BertPreTrainedModel,
            AdamW, get_linear_schedule_with_warmup,
            TrainingArguments,AutoTokenizer
        )

from sys import getsizeof
from retrieval import *

class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output

dataset = load_from_disk("../data/test_dataset")

f = Features(
        {
            "context": Value(dtype="string", id=None),
            "id": Value(dtype="string", id=None),
            "question": Value(dtype="string", id=None),
        }
    )

Targs = TrainingArguments(
    output_dir="dense_retrieval",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01
)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_checkpoint = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
p_encoder_dir, q_encoder_dir = "../p_encoder_dir", "../q_encoder_dir/" 

if p_encoder_dir == None:
    p_encoder_dir = model_checkpoint            
if q_encoder_dir == None:
    q_encoder_dir = model_checkpoint

p_encoder = BertEncoder.from_pretrained(p_encoder_dir).to(Targs.device)
q_encoder = BertEncoder.from_pretrained(q_encoder_dir).to(Targs.device)


retriever = DenseRetrieval(
    args = Targs,
    dataset = dataset,
    num_neg = 2,
    tokenizer = tokenizer,
    p_encoder = p_encoder,
    q_encoder = q_encoder,
    wiki_path = "../wikipedia_documents.json",
)

retriever.get_dense_embedding()
df = retriever.retrieve(dataset["validation"], topk=3)
print(df)