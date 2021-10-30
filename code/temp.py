# from elastic_search import ElasticSearchRetrieval
# from utils_qa import get_args
# from datasets import load_from_disk, concatenate_datasets
# import pickle
#
# model_args, data_args, training_args = get_args()
#
#
# retriever = ElasticSearchRetrieval(data_args)
#
# if data_args.elastic_index_name == 'wiki-index':
#     org_dataset = load_from_disk('../data/train_dataset')
# else:
#     with open('../data/preprocess_train.pkl', 'rb') as f:
#         org_dataset = pickle.load(f)
#
#
# if for_train:
#     full_ds = concatenate_datasets(
#             [
#                 org_dataset["train"].flatten_indices(),
#             ]
#         )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
#
# else:
#     full_ds = concatenate_datasets(
#             [
#                 org_dataset["train"].flatten_indices(),
#                 org_dataset["validation"].flatten_indices(),
#             ]
#         )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
#
#
# context_list, score_list = retriever.elastic_retrieval(full_ds, topk=k)
c = 51

def add(a, b=c if c == 0 else 5):
    return a + b

print(add(100))