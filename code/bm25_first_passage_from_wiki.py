import json
from tqdm import tqdm
from itertools import islice
from elastic_search import ElasticSearchRetrieval


class GetBM25:
    def __init__(self, data_args):
        with open("/opt/ml/data/wikipedia_documents.json", "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.corpus = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        self.retriever = ElasticSearchRetrieval(data_args)

    def get_bm25_wrong_passage(self, queries, context_dict, batch_size):

        return_passage = []
        print(f"total query number: {len(queries)}")
        for i, query in tqdm(enumerate(queries), desc="finding wrong bm25_passage", total=len(queries)):
            context_list, _, _ = self.retriever.elastic_retrieval(query, 1000)
            start_batch_idx = i // batch_size
            if i % 200 == 0:
                print(f"{i}번 째 query 돌파!")

            for passage in context_list:
                batch_dict = dict(islice(context_dict.items(), start_batch_idx * batch_size, (start_batch_idx + 1) * batch_size))
                if passage not in batch_dict:
                    return_passage.append(passage)
                    break
        return return_passage

    def get_bm25_passage(self, question_texts, topk):
        doc_scores = []
        doc_indices = []
        doc_contexts = []
        for question in tqdm(question_texts, desc="Get relevant passage by Elasticsearch", total=len(question_texts)):
            context_list, score_list, id_list = self.retriever.elastic_retrieval(question, topk)
            doc_contexts.append(context_list)
            doc_scores.append(score_list)
            doc_indices.append(id_list)

        return doc_contexts, doc_scores, doc_indices



