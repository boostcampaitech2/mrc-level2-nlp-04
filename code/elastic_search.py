import os
import json
import time
from contextlib import contextmanager

import pandas as pd
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

from datasets import load_from_disk
from prepare_dataset import make_custom_dataset


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class ElasticSearchRetrieval:
    def __init__(self, data_args, use_negative_sampling=False):
        self.data_args = data_args
        self.use_negative_sampling = use_negative_sampling
        self.index_name = data_args.elastic_index_name
        self.k = data_args.top_k_retrieval

        self.es = Elasticsearch()

        if not self.es.indices.exists(self.index_name):
            self.qa_records, self.wiki_articles = self.set_datas()
            self.set_index()
            self.populate_index(es_obj=self.es,
                                index_name=self.index_name,
                                evidence_corpus=self.wiki_articles)

    def set_datas(self):
        """elastic search 에 저장하는 데이터 세팅과정"""

        if not os.path.isfile('../data/preprocess_train.pkl'):
            make_custom_dataset('../data/preprocess_train.pkl')

        train_file = load_from_disk('../data/train_dataset')['train']
        # validation_file = load_from_disk('../data/train_dataset')['validation']

        if self.data_args.elastic_index_name == 'wiki-index':
            dataset_path = '../data/wikipedia_documents.json'
        elif self.data_args.elastic_index_name == 'preprocess-wiki-index':
            dataset_path = '../data/preprocess_wiki.json'

        if not os.path.isfile(dataset_path):
            print(dataset_path)
            make_custom_dataset(dataset_path)

        with open(dataset_path, 'r') as f:
            wiki = json.load(f)
        wiki_contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))

        qa_records = [{'example_id': train_file[i]['id'],
                       'document_title': train_file[i]['title'],
                       'question_text': train_file[i]['question'],
                       'answer': train_file[i]['answers']}
                      for i in range(len(train_file))]
        wiki_articles = [{'document_text': wiki_contexts[i]} for i in range(len(wiki_contexts))]

        return qa_records, wiki_articles

    def set_index(self):
        """index 생성 과정"""
        index_config = {
            'settings': {
                'analysis': {
                    'analyzer': {
                        'nori_analyzer': {
                            'type': 'custom',
                            'tokenizer': 'nori_tokenizer',
                            'decompound_mode': 'mixed',
                            'filter': ['shingle'],
                        }
                    }
                }
            },
            'mappings': {
                'dynamic': 'strict',
                'properties': {
                    'document_text': {
                        'type': 'text',
                        'analyzer': 'nori_analyzer',
                    }
                }
            }
        }

        print('elastic search ping:', self.es.ping())
        print(self.es.indices.create(index=self.data_args.elastic_index_name, body=index_config, ignore=400))

    def populate_index(self, es_obj, index_name, evidence_corpus):
        """
        생성된 elastic search 의 index_name 에 context 를 채우는 과정
        populate : 채우다
        """

        for i, rec in enumerate(tqdm(evidence_corpus)):
            try:
                es_obj.index(index=index_name, id=i, document=rec)
            except:
                print(f'Unable to load document {i}.')

        n_records = es_obj.count(index=index_name)['count']
        print(f'Succesfully loaded {n_records} into {index_name}')

    def retrieve(self, query_or_dataset, topk=None):
        if topk is not None:
            self.k = topk

        total = []
        scores = []

        with timer("query exhaustive search"):
            pbar = tqdm(query_or_dataset, desc='elastic search - question: ')
            for idx, example in enumerate(pbar):
                # top-k 만큼 context 검색
                context_list, score_list = self.elastic_retrieval(example['question'])

                if self.use_negative_sampling:
                    tmp = {
                        'question': example['question'],
                        'id': example['id'],
                        'original_context' : example['context'],
                        'context': context_list
                    }

                else:
                    tmp = {
                        'question': example['question'],
                        'id': example['id'],
                        'context': ' '.join(context_list)
                    }

                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]

                total.append(tmp)
                scores.append(sum(score_list))

            df = pd.DataFrame(total)

        return df, scores

    def elastic_retrieval(self, question_text):
        result = self.search_es(question_text)
        # 매칭된 context만 list형태로 만든다.
        context_list = [hit['_source']['document_text'] for hit in result['hits']['hits']]
        score_list = [hit['_score'] for hit in result['hits']['hits']]
        return context_list, score_list

    def search_es(self, question_text):
        query = {
            'query': {
                'match': {
                    'document_text': question_text
                }
            }
        }
        result = self.es.search(index=self.index_name, body=query, size=self.k)
        return result
