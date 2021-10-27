from copy import Error
from rank_bm25 import BM25Okapi
import os
import numpy as np
import json
from transformers import AutoTokenizer
from tqdm import tqdm
from itertools import islice

def get_bm25_wrong_passage(queries, context_dict, batch_size):
    with open("/opt/ml/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
    tokenized_corpus = []
    for x in tqdm(corpus, desc="tokenizing wiki corpus"):
        tokenized_corpus.append(tokenizer.tokenize(x))

    print(f"len of corpus: {len(corpus)}")

    bm25 = BM25Okapi(tokenized_corpus)

    queries = queries

    return_passage = []
    print(f"total query number: {len(queries)}")
    for i, query in tqdm(enumerate(queries), desc="finding bm25_passage", total=len(queries)):
        start_batch_idx = i // batch_size
        tokenized_query = tokenizer.tokenize(query)
        if i % 200 == 0:
            print(f"{i}번 째 query 돌파!")

        for passage in bm25.get_top_n(tokenized_query, corpus, n=len(corpus)):
            batch_dict = dict(islice(context_dict.items(), start_batch_idx * batch_size, (start_batch_idx + 1) * batch_size))
            if passage not in batch_dict:
                return_passage.append(passage)
                break
    return return_passage


def get_bm25_passage(queries, topk):
    with open("/opt/ml/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
        wiki = json.load(f)
    corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
    tokenized_corpus = []
    for x in tqdm(corpus, desc="tokenizing wiki corpus"):
        tokenized_corpus.append(tokenizer.tokenize(x))

    print(f"len of corpus: {len(corpus)}")

    bm25 = BM25Okapi(tokenized_corpus)

    queries = queries

    doc_scores = []
    doc_indices = []
    print(f"total query number: {len(queries)}")
    for i, query in tqdm(enumerate(queries), desc="finding bm25_passage", total=len(queries)):
        tokenized_query = tokenizer.tokenize(query)

        sim_scores = bm25.get_scores(tokenized_query)
        sim_scores = np.array(sim_scores)
        sorted_result = np.argsort(-sim_scores).tolist()
        topk_scores = sim_scores[sorted_result][:topk]
        topk_indices = sorted_result[:topk]
        doc_scores.append(topk_scores)
        doc_indices.append(topk_indices)
    return doc_scores, doc_indices
