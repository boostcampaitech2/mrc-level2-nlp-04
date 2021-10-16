import logging
import os
import random
import sys

from typing import List, Callable, NoReturn, NewType, Any
import dataclasses

import numpy as np
import torch
import wandb
from datasets import load_metric, load_from_disk, Dataset, DatasetDict
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer, AdamW, \
    get_linear_schedule_with_warmup

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece

from data_processing import DataProcessor
from utils_qa import postprocess_qa_predictions, check_no_error, AverageMeter
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)

# avoid huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
# Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_args():
    '''
    훈련 시 입력한 각종 Argument를 반환하는 함수
    '''
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


def set_seed_everything(seed):
    '''Random Seed를 고정하는 함수'''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

    return None


def get_models(model_args):
    model_config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        # 'use_fast' argument를 True로 설정할 경우 rust로 구현된 tokenizer를 사용할 수 있습니다.
        # False로 설정할 경우 python으로 구현된 tokenizer를 사용할 수 있으며,
        # rust version이 비교적 속도가 빠릅니다.
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=model_config,
    )

    return tokenizer, model_config, model


def get_data(training_args, model_args, data_args, tokenizer):
    '''train과 validation의 dataloader와 dataset를 반환하는 함수'''
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    train_dataset = datasets['train']
    valid_dataset = datasets['validation']

    train_column_names = train_dataset.column_names
    valid_column_names = valid_dataset.column_names

    data_processor = DataProcessor(tokenizer, model_args, data_args)
    train_dataset = data_processor.train_tokenizer(train_dataset, train_column_names)
    valid_dataset = data_processor.valid_tokenizer(valid_dataset, valid_column_names)
    valid_dataset_for_model = valid_dataset.remove_columns(['example_id', 'offset_mapping'])

    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=(8 if training_args.fp16 else None)
    )
    train_loader = DataLoader(train_dataset, collate_fn=data_collator,
                              batch_size=training_args.per_device_train_batch_size)

    valid_loader = DataLoader(valid_dataset_for_model, collate_fn=data_collator,
                              batch_size=training_args.per_device_eval_batch_size)

    return datasets, train_loader, valid_loader, train_dataset, valid_dataset


def get_optimizers(model, train_loader, training_args):
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, eps=training_args.adam_epsilon)
    scaler = GradScaler()
    num_training_steps = len(train_loader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=training_args.warmup_steps,
                                                num_training_steps=num_training_steps)

    return optimizer, scaler, scheduler


def train_per_step(model, optimizer, scaler, batch, training_args):
    """
    매 step 마다 학습을 하는 함수
    """
    model.train()
    with autocast():
        batch = batch.to(training_args.device)
        outputs = model(**batch)

        # output 안에 loss 가 들어있는 형태
        loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return loss.item()


def validation_per_steps(epoch, model, datasets, valid_loader, valid_dataset, training_args, model_args, data_args):
    """
    매 logging_step 마다 검증을 하는 함수
    """
    metric = load_metric('squad')

    model.eval()
    all_start_logits = []
    all_end_logits = []

    for batch in valid_loader:
        batch = batch.to(training_args.device)
        outputs = model(**batch)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        all_start_logits.append(start_logits.detach().cpu().numpy())
        all_end_logits.append(end_logits.detach().cpu().numpy())

    max_len = max(x.shape[1] for x in all_start_logits)

    start_logits_concat = create_and_fill_np_array(all_start_logits, valid_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, valid_dataset, max_len)

    del all_start_logits
    del all_end_logits

    valid_dataset.set_format(type=None, columns=list(valid_dataset.features.keys()))
    output_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(datasets['validation'], valid_dataset, output_numpy, datasets,
                                          training_args, data_args)
    valid_metrics = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)

    return valid_metrics


# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa_beam_search_no_trainer.py
def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float32)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]
        if step + batch_size < len(dataset):
            logits_concat[step: step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat


def post_processing_function(examples, features, predictions, datasets, training_args, data_args):
    # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        max_answer_length=data_args.max_answer_length,
        output_dir=training_args.output_dir,
    )
    # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [
        {"id": k, "prediction_text": v} for k, v in predictions.items()
    ]
    if training_args.do_predict:
        return formatted_predictions

    references = [
        {"id": ex["id"], "answers": ex['answers']}
        for ex in datasets["validation"]
    ]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def train_mrc(
        model,
        optimizer,
        scaler,
        scheduler,
        datasets,
        train_loader,
        valid_loader,
        train_dataset,
        valid_dataset,
        training_args,
        model_args,
        data_args,
        tokenizer
) -> NoReturn:
    """
    train & validation 함수
    """
    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    prev_f1 = 0
    prev_em = 0
    global_step = 0
    train_loss = AverageMeter()

    train_iterator = trange(int(training_args.num_train_epochs), desc='Epoch')
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator):
            # training phase
            loss = train_per_step(model, optimizer, scaler, batch, training_args)
            train_loss.update(loss, len(batch['input_ids']))
            global_step += 1
            description = f"{epoch + 1} epoch {global_step:>5d} step | loss: {train_loss.avg:.4f} | f1: {prev_f1:.4f} | best_em: {prev_em:.4f}"
            epoch_iterator.set_description(description)
            if scheduler is not None:
                scheduler.step()

            # validation phase
            if global_step % training_args.logging_steps == 0:
                with torch.no_grad():
                    valid_metrics = validation_per_steps(epoch, model, datasets, valid_loader, valid_dataset,
                                                         training_args, model_args, data_args)
                if valid_metrics['exact_match'] > prev_em:
                    torch.save(model, os.path.join(training_args.output_dir, f'{training_args.run_name}.pt'))
                    prev_em = valid_metrics['exact_match']
                    prev_f1 = valid_metrics['f1']
                wandb.log({
                    'train/loss': train_loss.avg,
                    'train/learning_rate':
                        scheduler.get_last_lr()[0] if scheduler is not None else training_args.learning_rate,
                    'eval/exact_match': valid_metrics['exact_match'],
                    'eval/f1_score': valid_metrics['f1'],
                    'global_step': global_step,
                })
                train_loss.reset()
            else:
                wandb.log({'global_step': global_step})


# TODO model_test 위해 만들었으니 테스트 끝나고 지울 것
# def main():
def main(project_name=None, model_name_or_path=None):
    model_args, data_args, training_args = get_args()

    if model_name_or_path is not None:
        model_args.model_name_or_path = model_name_or_path

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed_everything(training_args.seed)

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    training_args.output_dir = os.path.join(training_args.output_dir, project_name, model_name_or_path.split('/')[-1])
    training_args.per_device_train_batch_size = 16
    training_args.per_device_eval_batch_size = 128
    training_args.num_train_epochs = 10
    training_args.learning_rate = 3e-5
    training_args.warmup_steps = 1000
    training_args.weight_decay = 0.01
    training_args.logging_steps = 100
    training_args.fp16 = True
    training_args.project_name = project_name
    print(training_args)
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    tokenizer, model_config, model = get_models(model_args)
    datasets, train_loader, valid_loader, train_dataset, valid_dataset = get_data(training_args, model_args,
                                                                                  data_args, tokenizer)
    optimizer, scaler, scheduler = get_optimizers(model, train_loader, training_args)
    model.cuda()

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    if not os.path.isdir(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    # wandb setting
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = "true"

    wandb.init(project=training_args.project_name,
               name=model_args.model_name_or_path,
               entity='ssp',
               reinit=True,
               )

    train_mrc(model, optimizer, scaler, scheduler, datasets, train_loader, valid_loader, train_dataset,
              valid_dataset, training_args, model_args, data_args, tokenizer)
    wandb.join()


if __name__ == "__main__":
    # TODO test 를 위한 세팅이므로 나중엔 원래대로 돌려놓을 것 처음 상태는 main() 이거 하나밖에 없음
    # main()
    test = {
        'model_test': ['klue/roberta-small', 'klue/roberta-base', 'klue/roberta-large', 'klue/bert-base',
                       'monologg/koelectra-base-v3-discriminator']
    }

    for k, v in test.items():
        for model_name in v:
            main(project_name=k, model_name_or_path=model_name)
