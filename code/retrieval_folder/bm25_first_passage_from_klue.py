from rank_bm25 import BM25Okapi
import os
import json
from transformers import AutoTokenizer
import numpy as np
from datasets import load_from_disk


try:
    dataset = load_from_disk("mrc-level2-nlp-04/data/train_dataset")["train"]
except:
    dataset = load_from_disk("../../data/train_dataset")["train"]


corpus = list(set([example for example in dataset["context"]]))

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
tokenized_corpus = []
for x in corpus:
    tokenized_corpus.append(tokenizer.tokenize(x))

print(f"len of corpus: {len(corpus)}")

bm25 = BM25Okapi(tokenized_corpus)


x = 10
for _ in range(1):
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    tokenized_query = tokenizer.tokenize(query)
    doc_scores = bm25.get_scores(tokenized_query)

    print(doc_scores)
    print(len(doc_scores))

    print(bm25.get_top_n(tokenized_query, corpus, n=x))