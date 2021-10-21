from copy import Error
from rank_bm25 import BM25Okapi
import os
import json
from transformers import AutoTokenizer
from tqdm import tqdm


def get_bm25_passage(queries, ans_seqs):

    with open("/opt/ml/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
                wiki = json.load(f)
    corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")
    tokenizer.model_max_length = 30000
    tokenized_corpus = []
    for x in tqdm(corpus, desc="tokenizing wiki corpus"):
        tokenized_corpus.append(tokenizer.tokenize(x))

    print(f"len of corpus: {len(corpus)}")

    bm25 = BM25Okapi(tokenized_corpus)


    queries = queries

    return_passage = []
    print(f"total query number: {len(queries)}")
    for i, (query, answer) in tqdm(enumerate(zip(queries, ans_seqs)), desc="finding bm25_passage", total=len(queries)):
        tokenized_query = tokenizer.tokenize(query)
        if i % 200 == 0:
            print(f"{i}번 째 query를 진행중입니다")

        for passage in bm25.get_top_n(tokenized_query, corpus, n=len(corpus)):
            if answer not in passage:
                return_passage.append(passage)
                break
        else:
            print(f"{i}번 째 question의 passage에 모두 answer가 있습니다")
            print(passage)
            break
    return return_passage