"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys
import os
from tqdm import tqdm

from utils_qa import get_args, set_seed_everything, get_models, get_data, create_and_fill_np_array, \
    post_processing_function

logger = logging.getLogger(__name__)

# avoid huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
# Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    # get arguments
    model_args, data_args, training_args = get_args()
    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed_everything(training_args.seed)

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # get_tokenizer, model
    tokenizer, model = get_models(training_args, model_args)
    # load data
    datasets, test_loader, test_dataset = get_data(training_args, model_args, data_args, tokenizer)
    model.cuda()

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    if not os.path.isdir(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    inference(model, datasets, test_loader, test_dataset, training_args, data_args)


def inference(model, datasets, test_loader, test_dataset, training_args, data_args):
    model.eval()

    all_start_logits = []
    all_end_logits = []

    logger.info("*** Inference ***")

    test_iteratior = tqdm(test_loader, desc='Iteration', position=0, leave=True)
    for step, batch in enumerate(test_iteratior):
        batch = batch.to(training_args.device)
        outputs = model(**batch)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        all_start_logits.append(start_logits.detach().cpu().numpy())
        all_end_logits.append(end_logits.detach().cpu().numpy())

    max_len = max(x.shape[1] for x in all_start_logits)

    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

    del all_start_logits
    del all_end_logits

    test_dataset.set_format(type=None, columns=list(test_dataset.features.keys()))
    output_numpy = (start_logits_concat, end_logits_concat)
    post_processing_function(datasets['validation'], test_dataset, output_numpy, datasets,
                             training_args, data_args)


if __name__ == "__main__":
    main()
