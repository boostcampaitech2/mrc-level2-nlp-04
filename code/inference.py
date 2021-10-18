"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""

import logging
import sys
from typing import NoReturn

from utils_qa import check_no_error, compute_metrics, post_processing_function, get_args, \
    set_seed_everything, get_models, get_data
from trainer_qa import QuestionAnsweringTrainer

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def main():
    # get arguments
    model_args, data_args, training_args = get_args()

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed_everything(training_args.seed)

    # get_tokenizer, model
    tokenizer, model = get_models(training_args, model_args)

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, tokenizer, model)


def run_mrc(
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        model_args: ModelArguments,
        tokenizer,
        model,
) -> NoReturn:
    # load data
    datasets, eval_dataset, data_collator = get_data(training_args, model_args, data_args, tokenizer)

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    data_args.max_seq_length = max_seq_length

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()
