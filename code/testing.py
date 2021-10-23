import os
import json
from datasets import load_from_disk
import time
import faiss
import pickle
import numpy as np
import pandas as pd

import argparse
import hashlib 
import pprint

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union
from torch.utils.data import DataLoader, TensorDataset

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

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
from transformers.models.auto.configuration_auto import AutoConfig
    

from sys import getsizeof
from retrieval import *

def main(args):
    Targs = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01
    )
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_checkpoint = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    if args.is_train:
        p_encoder_dir, q_encoder_dir = model_checkpoint, model_checkpoint
        dataset = load_from_disk(args.train_dataset_dir)
    else:
        dataset = load_from_disk(args.test_dataset_dir)
        p_encoder_dir, q_encoder_dir = args.p_encoder_dir, args.q_encoder_dir
        
        f = Features(
                {
                    "context": Value(dtype="string", id=None),
                    "id": Value(dtype="string", id=None),
                    "question": Value(dtype="string", id=None),
                }
            )
    model_config = AutoConfig.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder(name=p_encoder_dir, config=model_config).to(Targs.device) #.from_pretrained(p_encoder_dir).to(Targs.device)
    q_encoder = BertEncoder(name=q_encoder_dir, config=model_config).to(Targs.device) #.from_pretrained(q_encoder_dir).to(Targs.device)

    retriever = DenseRetrieval(
        args = Targs,
        dataset = dataset['train'] if args.is_train else None, # train을 위한 dataset
        num_neg = 2,
        tokenizer = tokenizer,
        p_encoder = p_encoder,
        q_encoder = q_encoder,
        wiki_path = args.wiki_path,
        is_train = args.is_train
    )

    if args.is_train:
        retriever.train()
        p_encoder.save_pretrained(args.p_encoder_dir)
        q_encoder.save_pretrained(args.q_encoder_dir)
    else:     
        retriever.get_dense_embedding()
        df, uni_set = retriever.retrieve(dataset["validation"], topk=10)
        # df_datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
        print(uni_set)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--p_encoder_dir", default="../p_encoder_dir", type=str, help=""
    )
    parser.add_argument(
        "--q_encoder_dir", default="../q_encoder_dir", type=str, help=""
    )
    parser.add_argument(
        "--wiki_path", default="../wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument(
        "--train_dataset_dir", default="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--test_dataset_dir", default="../data/test_dataset", type=str, help=""
    )
    parser.add_argument(
        "--is_train", default=False, type=bool, help=""
    )
    args = parser.parse_args()
    main(args)