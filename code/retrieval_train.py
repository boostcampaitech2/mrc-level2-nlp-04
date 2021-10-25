import argparse
import logging
import os.path
import sys

import torch
import torch.nn.functional as F
import wandb
from knockknock import slack_sender
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AdamW, TrainingArguments

from dense_retrieval import DenseRetrieval
from utils_retrieval import get_encoders
from utils_qa import set_seed_everything

logger = logging.getLogger(__name__)

webhook_url = "https://hooks.slack.com/services/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"


@slack_sender(webhook_url=webhook_url, channel="#test_for_knock_knock")
def main(args):
    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed_everything(args.seed)

    # get_tokenizer, model
    tokenizer, p_encoder, q_encoder = get_encoders(args)
    if torch.cuda.is_available():
        p_encoder.to('cuda')
        q_encoder.to('cuda')

    training_args = TrainingArguments(output_dir=args.output_dir,
                                      evaluation_strategy='epoch',
                                      learning_rate=args.learning_rate,
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                      num_train_epochs=args.epoch,
                                      weight_decay=0.01)

    # set wandb
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=args.project_name, entity='ssp', name=args.run_name, reinit=True)

    best_acc = train_retrieval(training_args, args, tokenizer, p_encoder, q_encoder)


    wandb.join()

    return {'best_acc': best_acc}


def training_per_step(training_args, args, batch, p_encoder, q_encoder, criterion, scaler):
    with autocast():
        batch_loss, batch_acc = 0, 0

        if torch.cuda.is_available():
            batch = tuple(t.cuda() for t in batch)

        if 'roberta' in args.model_name_or_path:
            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1]}
            q_inputs = {'input_ids': batch[2],
                        'attention_mask': batch[3]}
        else:
            p_inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]}
            q_inputs = {'input_ids': batch[3],
                        'attention_mask': batch[4],
                        'token_type_ids': batch[5]}

        p_outputs = p_encoder(**p_inputs)  # (batch_size, emb_dim)
        q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

        # Calculate similarity score & loss
        sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, batch_size)

        # target : position of positive samples = diagonal element
        targets = torch.arange(0, training_args.per_device_train_batch_size).long()
        if torch.cuda.is_available():
            targets = targets.to('cuda')

        sim_scores = F.log_softmax(sim_scores, dim=1)
        _, preds = torch.max(sim_scores, dim=1)

        loss = criterion(sim_scores, targets)
        scaler.scale(loss).backward()

        batch_loss += loss.cpu().item()
        batch_acc += torch.sum(preds.cpu() == targets.cpu())

    return p_encoder, q_encoder, batch_loss, batch_acc


def evaluating_per_step(training_args, args, batch, p_encoder, q_encoder):
    batch_acc = 0

    if torch.cuda.is_available():
        batch = tuple(t.cuda() for t in batch)

    if 'roberta' in args.model_name_or_path:
        p_inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1]}
        q_inputs = {'input_ids': batch[2],
                    'attention_mask': batch[3]}
    else:
        p_inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]}
        q_inputs = {'input_ids': batch[3],
                    'attention_mask': batch[4],
                    'token_type_ids': batch[5]}

    p_outputs = p_encoder(**p_inputs)  # (batch_size, emb_dim)
    q_outputs = q_encoder(**q_inputs)  # (batch_size, emb_dim)

    # Calculate similarity score & loss
    sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))
    sim_scores = F.log_softmax(sim_scores, dim=1)
    _, preds = torch.max(sim_scores, dim=1)

    # target : position of positive samples = diagonal element
    targets = torch.arange(0, training_args.per_device_eval_batch_size).long()
    targets = targets

    batch_acc += torch.sum(preds.cpu() == targets)

    return p_encoder, q_encoder, batch_acc


def train_retrieval(training_args, args, tokenizer, p_encoder, q_encoder):
    dense_retrieval = DenseRetrieval(training_args, args, tokenizer, p_encoder, q_encoder)

    train_dataloader, eval_dataloader = dense_retrieval.get_dataloader()

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
        {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": training_args.weight_decay},
        {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        eps=training_args.adam_epsilon
    )
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-6)
    criterion = nn.NLLLoss()

    # Start training!
    best_acc = 0.0

    train_iterator = tqdm(range(int(training_args.num_train_epochs)), desc='Epoch')
    for epoch in train_iterator:
        optimizer.zero_grad()
        p_encoder.zero_grad()
        q_encoder.zero_grad()
        torch.cuda.empty_cache()

        ## train phase
        running_loss, running_acc, num_cnt = 0, 0, 0

        p_encoder.train()
        q_encoder.train()

        epoch_iterator = tqdm(train_dataloader, desc='train-Iteration')
        for step, batch in enumerate(epoch_iterator):
            p_encoder, q_encoder, batch_loss, batch_acc = training_per_step(training_args, args, batch,
                                                                            p_encoder, q_encoder, criterion, scaler)
            running_loss += batch_loss / training_args.per_device_train_batch_size
            running_acc += batch_acc / training_args.per_device_train_batch_size
            num_cnt += 1

            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                log_step = epoch * len(epoch_iterator) + step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                p_encoder.zero_grad()
                q_encoder.zero_grad()

        train_epoch_loss = float(running_loss / num_cnt)
        train_epoch_acc = float((running_acc.double() / num_cnt).cpu() * 100)
        print(f'global step-{log_step} | Loss: {train_epoch_loss:.4f} Accuracy: {train_epoch_acc:.2f}')

        # eval phase
        epoch_iterator = tqdm(eval_dataloader, desc='valid-Iteration')
        p_encoder.eval()
        q_encoder.eval()

        running_acc, num_cnt = 0, 0
        for step, batch in enumerate(epoch_iterator):
            with torch.no_grad():
                p_encoder, q_encoder, batch_acc = \
                    evaluating_per_step(training_args, args, batch, p_encoder, q_encoder)

                running_acc += batch_acc / training_args.per_device_eval_batch_size
                num_cnt += 1

        eval_epoch_acc = float((running_acc / num_cnt) * 100)
        print(f'Epoch-{epoch} | Accuracy: {eval_epoch_acc:.2f}')

        if eval_epoch_acc > best_acc:
            best_epoch = epoch
            best_acc = eval_epoch_acc

            p_save_path = os.path.join(training_args.output_dir, 'p_encoder')
            q_save_path = os.path.join(training_args.output_dir, 'q_encoder')
            if not os.path.exists(p_save_path):
                os.makedirs(p_save_path, exist_ok=True)
            if not os.path.exists(q_save_path):
                os.makedirs(q_save_path, exist_ok=True)

            p_encoder.save_pretrained(p_save_path)
            q_encoder.save_pretrained(q_save_path)
            print(f'\t===> best model saved - {best_epoch} / Accuracy: {best_acc:.2f}')

        wandb.log({
            'train/loss': train_epoch_loss,
            'train/learning_rate': scheduler.get_last_lr()[0],
            'eval/epoch_acc': eval_epoch_acc,
            'epoch': epoch,
        })

    return best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='../retrieval_output/')
    parser.add_argument('--dataset_name', type=str, default='../data/train_dataset')
    parser.add_argument('--model_name_or_path', type=str, default='klue/roberta-small')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--run_name', type=str, default='roberta-small')
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--use_trained_model', type=bool, default=False)
    parser.add_argument('--use_custom', type=bool, default=False)

    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.run_name)

    main(args)
