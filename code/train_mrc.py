import logging
import os
import sys
import random

import torch
import wandb
from datasets import load_from_disk, Dataset, DatasetDict, tqdm
from sklearn.model_selection import KFold
from torch.cuda.amp import autocast, GradScaler
from tqdm import trange
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from utils_qa import get_args, set_seed_everything, logger, get_models, check_no_error, get_data, \
    make_combined_dataset, AverageMeter, create_and_fill_np_array, post_processing_function, metric, custom_to_mask


def get_dataloader(training_args, train_dataset, eval_dataset, data_collator):
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator,
                                  batch_size=training_args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset.remove_columns(['example_id', 'offset_mapping']),
                                 collate_fn=data_collator,
                                 batch_size=training_args.per_device_eval_batch_size)

    return train_dataloader, eval_dataloader


def train_per_step(model, scaler, batch, tokenizer, training_args):
    """
    매 step 마다 학습을 하는 함수
    """
    model.train()
    with autocast():
        mask_props = 0.8
        mask_p = random.random()
        if mask_p < mask_props:
            # 확률 안에 들면 mask 적용
            batch = custom_to_mask(batch, tokenizer)

        batch = batch.to(training_args.device)

        outputs = model(**batch)
        loss = outputs.loss
        scaler.scale(loss).backward()

    return loss.item()


def validation_per_steps(model, datasets, valid_loader, valid_dataset, training_args):
    """
    매 logging_step 마다 검증을 하는 함수
    """
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
    prediction = post_processing_function(datasets['validation'], valid_dataset, output_numpy, training_args)
    valid_metrics = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)

    return valid_metrics


def train_mrc(training_args, tokenizer, model, datasets, train_dataset, eval_dataset, data_collator, k=0):
    """
    train & validation 함수
    """
    wandb.init(project=training_args.project_name,
               name=training_args.run_name + (f'_{k}' if k else ''),
               entity='ssp',
               reinit=True,
               )

    if not os.path.isdir(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    train_loader, eval_loader = get_dataloader(training_args, train_dataset, eval_dataset, data_collator)

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        eps=training_args.adam_epsilon
    )
    scaler = GradScaler()
    t_total = len(train_loader) // training_args.gradient_accumulation_steps * training_args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=t_total * training_args.warmup_ratio,
        num_training_steps=t_total
    )

    prev_f1 = 0
    prev_em = 0
    global_step = 0
    early_stop_cnt = 0
    train_loss = AverageMeter()

    train_iterator = trange(int(training_args.num_train_epochs), desc='Epoch')
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_loader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator):
            # training phase
            loss = train_per_step(model, scaler, batch, tokenizer, training_args)
            train_loss.update(loss, len(batch['input_ids']))
            global_step += 1
            description = f"{epoch + 1} epoch {global_step} step | loss: {train_loss.avg:.4f} | f1: {prev_f1:.4f} | best_em: {prev_em:.4f}"
            epoch_iterator.set_description(description)

            if (global_step) % training_args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            # validation phase
            if (global_step + 1) % (training_args.logging_steps * training_args.gradient_accumulation_steps) == 0:
                with torch.no_grad():
                    valid_metrics = validation_per_steps(model, datasets, eval_loader, eval_dataset, training_args)
                if valid_metrics['exact_match'] > prev_em:
                    model.save_pretrained(training_args.output_dir)
                    prev_em = valid_metrics['exact_match']
                    prev_f1 = valid_metrics['f1']
                    early_stop_cnt = 0
                else:
                    early_stop_cnt += 1
                wandb.log({
                    'train/loss': train_loss.avg,
                    'train/learning_rate':
                        scheduler.get_last_lr()[0] if scheduler is not None else training_args.learning_rate,
                    'eval/exact_match': valid_metrics['exact_match'],
                    'eval/f1_score': valid_metrics['f1'],
                    'eval/best_em': prev_em,
                    'global_step': global_step,
                })
                train_loss.reset()
            else:
                wandb.log({'global_step': global_step})

            if early_stop_cnt == 5:
                return


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
        model.to(training_args.device)

        # 오류가 있는지 확인합니다.
        last_checkpoint, max_seq_length = check_no_error(
            data_args, training_args, tokenizer
        )
        data_args.max_seq_length = max_seq_length

        datasets, train_dataset, eval_dataset, data_collator = get_data(training_args, model_args, data_args, tokenizer)
        # if "validation" not in datasets:
        #     raise ValueError("--do_eval requires a validation dataset")
        train_mrc(training_args, tokenizer, model, datasets, train_dataset, eval_dataset, data_collator)
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
            train_dataset, eval_dataset = map(Dataset.from_dict,
                                              [combined_datasets[train_index], combined_datasets[valid_index]])
            datasets = DatasetDict({'train': train_dataset, 'validation': eval_dataset})

            train_dataset = data_processor.train_tokenizer(train_dataset, train_dataset.column_names)
            eval_dataset = data_processor.valid_tokenizer(eval_dataset, eval_dataset.column_names)

            _, _, model = get_models(model_args)

            training_args.output_dir = origin_output_dir + f'/{idx}'
            print(f"####### start training on fold {idx} #######")
            train_mrc(training_args, tokenizer, model, datasets, train_dataset, eval_dataset, data_collator, idx)
            print(f"####### end training on fold {idx} #######")


if __name__ == "__main__":
    main()
