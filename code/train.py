import logging
import os
import sys

from typing import NoReturn

import wandb
from transformers import EarlyStoppingCallback

from utils_qa import check_no_error, get_args, set_seed_everything, get_models, get_data, \
    post_processing_function, compute_metrics, make_combined_dataset
from trainer_qa import QuestionAnsweringTrainer

from datasets import load_from_disk, DatasetDict, Dataset
from sklearn.model_selection import KFold

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    model_args, data_args, training_args = get_args()

    # do_train mrc model 혹은 do_eval mrc model
    if not (training_args.do_train or training_args.do_eval):
        print("####### set do_train or do_eval #######")
        return

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed_everything(training_args.seed)

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # wandb setting
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = "true"

    if training_args.fold is False:
        tokenizer, model_config, model = get_models(model_args)

        # 오류가 있는지 확인합니다.
        last_checkpoint, max_seq_length = check_no_error(
            data_args, training_args, tokenizer
        )
        data_args.max_seq_length = max_seq_length

        datasets, train_dataset, eval_dataset, data_collator = get_data(training_args, model_args, data_args, tokenizer)
        # if "validation" not in datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        run_mrc(data_args, training_args, model_args, tokenizer, model,
                datasets, train_dataset, eval_dataset, data_collator, last_checkpoint)
    else:
        from transformers import DataCollatorWithPadding
        from data_processing import DataProcessor

        tokenizer, model_config, _ = get_models(model_args)

        # 오류가 있는지 확인합니다.
        last_checkpoint, max_seq_length = check_no_error(
            data_args, training_args, tokenizer
        )
        data_args.max_seq_length = max_seq_length

        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=(8 if training_args.fp16 else None)
        )
        data_processor = DataProcessor(tokenizer, model_args, data_args)
        origin_output_dir = training_args.output_dir

        if not os.path.isdir('../data/combined_dataset'):
            make_combined_dataset(data_args, '../data/combined_dataset')
        if not os.path.isdir('../data/concat_combined_dataset'):
            make_combined_dataset(data_args, '../data/concat_combined_dataset')

        if data_args.dataset_name == 'concat':
            combined_datasets = load_from_disk('../data/concat_combined_dataset')
        else:
            combined_datasets = load_from_disk('../data/combined_dataset')

        kf = KFold(n_splits=5, random_state=training_args.seed, shuffle=True)
        for idx, (train_index, valid_index) in enumerate(kf.split(combined_datasets), 1):
            train_dataset, eval_dataset = map(Dataset.from_dict, [combined_datasets[train_index], combined_datasets[valid_index]])
            datasets = DatasetDict({'train' : train_dataset, 'validation' : eval_dataset})

            train_dataset = data_processor.train_tokenizer(train_dataset, train_dataset.column_names)
            eval_dataset = data_processor.valid_tokenizer(eval_dataset, eval_dataset.column_names)

            _, _, model = get_models(model_args)

            training_args.output_dir = origin_output_dir + f'/{idx}'
            print(f"####### start training on fold {idx} #######")
            run_mrc(data_args, training_args, model_args, tokenizer, model,
                    datasets, train_dataset, eval_dataset, data_collator, last_checkpoint, idx)
            print(f"####### end training on fold {idx} #######")

    print(f"####### Saved at {training_args.output_dir} #######")
    if training_args.with_inference:
        if training_args.fold:
            training_args.output_dir = origin_output_dir
        sub, output, project, run = training_args.output_dir.split('/')
        output = '/'.join([sub, output])

        if not training_args.fold:
            print(project, run)
            string = (
                f'python inference.py --do_predict --project_name {project} \
                --run_name {run}'
                + (f' --additional_model {model_args.additional_model}'
                   if model_args.additional_model is not None else '')
                + f' --top_k_retrieval {data_args.top_k_retrieval}'
            )
            print(f"####### inference start automatically #######")
            os.system(string)
        else:
            for k in range(1, 6):
                string = (
                        f"python inference.py --do_predict --project_name {project} \
                                --run_name {run}/{k}"
                        + (f" --additional_model {model_args.additional_model}"
                           if model_args.additional_model is not None else '')
                        + f' --top_k_retrieval {data_args.top_k_retrieval}'
                        + ' --elastic_index_name preprocess-wiki-index'
                )
                print(f"####### fold {k} inference start automatically #######")
                os.system(string)

            string = f'python combine.py --project_name {project} --run_name {run}'
            print(f"####### fold predictions start to be combined #######")
            os.system(string)
            print(print(f"####### combining end #######"))


def run_mrc(
        data_args: DataTrainingArguments,
        training_args: TrainingArguments,
        model_args: ModelArguments,
        tokenizer,
        model,
        datasets,
        train_dataset,
        eval_dataset,
        data_collator,
        last_checkpoint,
        k = 0,
) -> NoReturn:

    wandb.init(project=training_args.project_name,
               name=training_args.run_name + (f'_{k}' if k else ''),
               entity='ssp',
               reinit=True,
               )

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)],  # early stopping
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    wandb.join()

if __name__ == "__main__":
    main()
