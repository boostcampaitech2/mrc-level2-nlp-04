import os

import torch
import wandb
from datasets import load_from_disk, concatenate_datasets

from retrieval import SparseRetrieval
from dense_custom_retrieval import get_encoders, DenseRetrieval, timer
from utils_qa import get_args

if __name__ == "__main__":
    # get arguments
    model_args, data_args, training_args = get_args()

    # Test sparse
    org_dataset = load_from_disk(data_args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)




    tokenizer, p_encoder,  q_encoder = get_encoders(training_args, model_args)

    if torch.cuda.is_available():
        q_encoder.to('cuda')
        p_encoder.to('cuda')

    if data_args.eval_retrieval == 'sparse':
        retriever = SparseRetrieval(tokenize_fn=tokenizer.tokenize)
        # retriever.get_sparse_embedding() # 없어도 될 것 같은데
    elif data_args.eval_retrieval == 'dense':
        retriever = DenseRetrieval(training_args, model_args, data_args, tokenizer,p_encoder, q_encoder)
        # retriever.get_dense_embedding() # 없어도 될 것 같은데
    elif data_args.eval_retrieval == 'both':
        pass
    else:
        raise ValueError('data_args.eval_retrieval 을 sparse or dense or both 로 설정하세요!')

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    # wandb setting
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = "true"

    wandb.init(project=training_args.project_name,
               name=training_args.run_name,
               entity='ssp',
               reinit=True,
               )

    if data_args.use_faiss:

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve_faiss(full_ds)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    else:
        for k in [10, 30]:
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve_rerank(training_args, model_args, full_ds, topk=k)
                df["correct"] = df.apply(lambda x: x["original_context"] in x["context"], axis=1)
                accuracy = round(df['correct'].sum() / len(df) * 100, 2)
                print(
                    f"Top-{k}\n"
                    "correct retrieval result by exhaustive search",
                    accuracy,
                )

                wandb.log({
                    'k': k,
                    'accuracy': accuracy,
                })