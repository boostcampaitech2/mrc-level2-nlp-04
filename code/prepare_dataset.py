import json
import os
import pickle
import re

import pandas as pd
from datasets import Features, Sequence, Value, load_from_disk, DatasetDict, Dataset
from elasticsearch import Elasticsearch
from tqdm import tqdm


def preprocess(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r'#', ' ', text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9가-힣ㄱ-ㅎㅏ-ㅣぁ-ゔァ-ヴー々〆〤一-龥<>()\s\.\?!》《≪≫\'<>〈〉:‘’%,『』「」＜＞・\"-“”∧]", "", text)
    return text


def run_preprocess(data_dict):
    start_ids = data_dict["answers"]["answer_start"][0]
    before = data_dict["context"][:start_ids]
    after = data_dict["context"][start_ids:]
    process_before = preprocess(before)
    process_after = preprocess(after)
    process_data = process_before + process_after
    ids_move = len(before) - len(process_before)
    data_dict["context"] = process_data
    data_dict["answers"]["answer_start"][0] = start_ids - ids_move
    return data_dict


def run_preprocess_to_wiki(data_dict):
    context = data_dict["text"]
    process_data = preprocess(context)
    data_dict["text"] = process_data
    return data_dict


def save_data(data_path, new_wiki):
    with open(data_path, 'w', encoding='utf-8') as make_file:
        json.dump(new_wiki, make_file, indent="\t", ensure_ascii=False)


def save_pickle(save_path, data_set):
    file = open(save_path, "wb")
    pickle.dump(data_set, file)
    file.close()
    return None


def get_pickle(pickle_path):
    f = open(pickle_path, "rb")
    dataset = pickle.load(f)
    f.close()
    return dataset


def search_es(es_obj, index_name, question_text, n_results):
    query = {
            'query': {
                'match': {
                    'document_text': question_text
                    }
                }
            }
    res = es_obj.search(index=index_name, body=query, size=n_results)
    return res


def make_custom_dataset(dataset_path):
    if not (os.path.isdir('../data/train_dataset') or
            os.path.isdir('../data/wikipedia_documents.json')):
        raise Exception("Set the original data path to '../data'")

    train_f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None),
                                                     'answer_start': Value(dtype='int32', id=None)},
                                            length=-1,
                                            id=None),
                        'context': Value(dtype='string', id=None),
                        'id': Value(dtype='string', id=None),
                        'question': Value(dtype='string', id=None)})

    if not os.path.isfile('../data/preprocess_wiki.json'):
        with open('../data/wikipedia_documents.json', 'r') as f:
            wiki = json.load(f)

        new_wiki = dict()
        for ids in tqdm(range(len(wiki))):
            new_wiki[str(ids)] = run_preprocess_to_wiki(wiki[str(ids)])
        save_data('../data/preprocess_wiki.json', new_wiki)

    if not os.path.isfile('../data/preprocess_train.pkl'):
        train_dataset = load_from_disk('../data/train_dataset')['train']
        valid_dataset = load_from_disk('../data/train_dataset')['validation']

        new_train_data, new_valid_data = [], []
        for data in tqdm(train_dataset):
            new_data = run_preprocess(data)
            new_train_data.append(new_data)
        for data in tqdm(valid_dataset):
            new_data = run_preprocess(data)
            new_valid_data.append(new_data)

        train_df = pd.DataFrame(new_train_data)
        valid_df = pd.DataFrame(new_valid_data)
        dataset = DatasetDict({'train': Dataset.from_pandas(train_df, features=train_f),
                               'validation': Dataset.from_pandas(valid_df, features=train_f)})
        save_pickle(dataset_path, dataset)

        if 'preprocess' in dataset_path:
            return dataset

    if 'concat' in dataset_path:
        base_dataset = get_pickle('../data/preprocess_train.pkl')
        train_dataset, valid_dataset = base_dataset['train'], base_dataset['validation']

        train_data = [{'id': train_dataset[i]['id'],
                       'question': train_dataset[i]['question'],
                       'answers': train_dataset[i]['answers'],
                       'context': train_dataset[i]['context']}
                      for i in range(len(train_dataset))]
        valid_data = [{'id': valid_dataset[i]['id'],
                       'question': valid_dataset[i]['question'],
                       'answers': valid_dataset[i]['answers'],
                       'context': valid_dataset[i]['context']}
                      for i in range(len(valid_dataset))]

        es = Elasticsearch()

        k = 5  # k : how many contexts to concatenate
        for idx, train in enumerate(train_data):
            res = search_es(es, 'preprocess-wiki-index', train['question'], k)
            context_list = [(hit['_source']['document_text'], hit['_score']) for hit in res['hits']['hits']]
            contexts = train['context']
            count = 0
            for context in context_list:
                # if same context already exists, don't concatenate
                if train['context'] == context[0]:
                    continue
                contexts += ' ' + context[0]
                count += 1
                if count == (k - 1):
                    break
            train_data[idx]['context'] = contexts

        for idx, valid in enumerate(valid_dataset):
            result = search_es(es, 'preprocess-wiki-index', valid['question'], k)
            context_list = [(hit['_source']['document_text'], hit['_score']) for hit in result['hits']['hits']]
            contexts = valid['context']
            count = 0
            for context in context_list:
                if valid['context'] == context[0]:
                    continue
                contexts += ' ' + context[0]
                count += 1
                if count == (k - 1):
                    break
            valid_data[idx]['context'] = contexts

        train_df = pd.DataFrame(train_data)
        valid_df = pd.DataFrame(valid_data)
        dataset = DatasetDict({'train': Dataset.from_pandas(train_df, features=train_f),
                               'validation': Dataset.from_pandas(valid_df, features=train_f)})
        save_pickle(dataset_path, dataset)
        return dataset
