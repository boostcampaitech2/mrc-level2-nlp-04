from typing import Callable, List

from datasets import DatasetDict, Features, Value, Sequence, Dataset
from transformers import TrainingArguments

from arguments import DataTrainingArguments
from retrieval import SparseRetrieval
from elasticsearch_retrieval import *
import kss
import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util, models

def run_sparse_retrieval(
        tokenize_fn: Callable[[str], List[str]],
        datasets: DatasetDict,
        training_args: TrainingArguments,
        data_args: DataTrainingArguments,
        data_path: str = "../data",
        context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(
        tokenize_fn=tokenize_fn, data_path=data_path, context_path=context_path
    )
    retriever.get_sparse_embedding()

    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets

def run_elasticsearch(datasets, concat_num, model_args, is_sentence_trainformer):
    """
    run elasticsearch and filter sentences
    Args:
        datasets
        concat_num: number of texts to import from elasticsearch
        is_sentence_trainformer: whether sentence trainformer is used or not
    Returns:
        datasets: test data
        scores: elasticsearch scores
    """
    # elastic setting & load index
    es, index_name = elastic_setting(model_args.retrieval_elastic_index)
    # load sentence transformer model
    if is_sentence_trainformer:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = SentenceTransformer("Huffon/sentence-klue-roberta-base") #sentence분류이므로 그에 맞는 모델이 필요. 일반 모델 사용하면 sentence 분류 층을 하나 더 쌓아야함.
    question_texts = datasets["validation"]["question"]
    total = []
    scores = []

    pbar = tqdm(enumerate(question_texts), total=len(question_texts), position=0, leave=True)
    for step, question_text in pbar:
        # concat_num만큼 context 검색
        context_list = elastic_retrieval(es, index_name, question_text, concat_num)
        score = []
        concat_context = []

        if is_sentence_trainformer:
            # question embedding
            question_embedding = model.encode(question_text)
            # use sentence transformer
            for i in range(len(context_list)):
                temp_context = []
                # separate context by sentence
                for sent in kss.split_sentences(context_list[i][0]):
                    # question embedding과 sentence embedding의 cosine similarity 계산
                    # -0.2 보다 높은 sentence만 append
                    if util.pytorch_cos_sim(question_embedding, model.encode(sent))[0] > -0.2:
                        temp_context.append(sent)

                concat_context.append(" ".join(temp_context))
        else:
            # not use sentence transformer
            for i in range(len(context_list)):
                concat_context.append(context_list[i][0])

        tmp = {
            "question": question_text,
            "id": datasets["validation"]["id"][step],
            "context": " <SEP> ".join(concat_context) if is_sentence_trainformer else " ".join(concat_context)
        }

        score.append(context_list[0][1])
        total.append(tmp)
        scores.append(score)

    df = pd.DataFrame(total)
    f = Features({'context': Value(dtype='string', id=None),
                  'id': Value(dtype='string', id=None),
                  'question': Value(dtype='string', id=None)})
    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})

    return datasets, scores
