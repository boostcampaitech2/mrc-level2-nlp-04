import logging
import os.path
from collections import defaultdict
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer
from datasets import load_from_disk
from utils_qa import set_seed_everything, get_args
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from bm25_first_passage_from_wiki import get_bm25_wrong_passage
from dense_retrieval import get_encoders


def main():
    model_args, data_args, training_args = get_args()
    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed_everything(training_args.seed)

    tokenizer, p_encoder, q_encoder = get_encoders(training_args, model_args)

    if torch.cuda.is_available():
        p_encoder.to('cuda')
        q_encoder.to('cuda')

    # set wandb
    os.environ['WANDB_LOG_MODEL'] = 'true'
    os.environ['WANDB_WATCH'] = 'all'
    os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=training_args.project_name,
               entity='ssp',
               name=training_args.retrieval_run_name,
               reinit=True,
               )

    best_acc = train_retrieval(training_args, model_args, data_args, tokenizer, p_encoder, q_encoder)

    wandb.join()

    return {'best_acc': best_acc}


def get_dataloader(tokenizer, training_args, model_args, data_args):
    # train, validation, test의 dataloader와 dataset를 반환하는 함수
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)
    train_dataset = datasets['train']
    eval_dataset = datasets['validation']

    context_dict = defaultdict(int)
    contexts = train_dataset['context']
    for context in contexts:
        context_dict[context]

    bm25_passages = get_bm25_wrong_passage(train_dataset["question"], context_dict, batch_size=training_args.per_device_retrieval_train_batch_size)

    train_p_bm25_seqs = tokenizer(bm25_passages, padding="max_length", truncation=True, return_tensors='pt')

    train_q_seqs = tokenizer(
        train_dataset['question'], padding='max_length', truncation=True, return_tensors='pt',
        return_token_type_ids=False if 'roberta' in model_args.retrieval_model_name_or_path else True)
    train_p_seqs = tokenizer(
        train_dataset['context'], padding='max_length', truncation=True, return_tensors='pt',
        return_token_type_ids=False if 'roberta' in model_args.retrieval_model_name_or_path else True)
    eval_q_seqs = tokenizer(
        eval_dataset['question'], padding='max_length', truncation=True, return_tensors='pt',
        return_token_type_ids=False if 'roberta' in model_args.retrieval_model_name_or_path else True)
    eval_p_seqs = tokenizer(
        eval_dataset['context'], padding='max_length', truncation=True, return_tensors='pt',
        return_token_type_ids=False if 'roberta' in model_args.retrieval_model_name_or_path else True)

    if 'roberta' in model_args.retrieval_model_name_or_path:
        train_dataset = TensorDataset(train_p_seqs['input_ids'], train_p_seqs['attention_mask'],
                                      train_p_bm25_seqs['input_ids'], train_p_bm25_seqs['attention_mask'],
                                      train_q_seqs['input_ids'], train_q_seqs['attention_mask'])
        eval_dataset = TensorDataset(eval_p_seqs['input_ids'], eval_p_seqs['attention_mask'],
                                     eval_q_seqs['input_ids'], eval_q_seqs['attention_mask'])
    else:
        train_dataset = TensorDataset(
            train_p_seqs['input_ids'], train_p_seqs['attention_mask'], train_p_seqs['token_type_ids'],
            train_p_bm25_seqs['input_ids'], train_p_bm25_seqs['attention_mask'], train_p_bm25_seqs['token_type_ids'],
            train_q_seqs['input_ids'], train_q_seqs['attention_mask'], train_q_seqs['token_type_ids'])
        eval_dataset = TensorDataset(
            eval_p_seqs['input_ids'], eval_p_seqs['attention_mask'], eval_p_seqs['token_type_ids'],
            eval_q_seqs['input_ids'], eval_q_seqs['attention_mask'], eval_q_seqs['token_type_ids'])

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=training_args.per_device_retrieval_train_batch_size)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=training_args.per_device_retrieval_eval_batch_size)

    return train_dataloader, eval_dataloader


def training_per_step(training_args, model_args, batch, p_encoder, q_encoder, criterion, scaler):
    with autocast():
        batch_loss, batch_acc = 0, 0
        if torch.cuda.is_available():
            batch = tuple(t.cuda() for t in batch)

        if 'roberta' in model_args.retrieval_model_name_or_path:
            p_inputs = {'input_ids': torch.cat((batch[0], batch[2]), dim=0),
                        'attention_mask': torch.cat((batch[1], batch[3]), dim=0)}
            q_inputs = {'input_ids': batch[4],
                        'attention_mask': batch[5]}
        else:
            p_inputs = {'input_ids': torch.cat((batch[0], batch[3]), dim=0),
                        'attention_mask': torch.cat((batch[1], batch[4]), dim=0),
                        'token_type_ids': torch.cat((batch[2], batch[5]), dim=0)}
            q_inputs = {'input_ids': batch[6],
                        'attention_mask': batch[7],
                        'token_type_ids': batch[8]}

        p_outputs = p_encoder(**p_inputs)  # (2*batch_size, hidden_dim)
        q_outputs = q_encoder(**q_inputs)  # (batch_size, hidden_dim)

        # Calculate similarity score & loss
        sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, 2*batch_size)

        # target : position of positive samples = diagonal element
        targets = torch.arange(0, sim_scores.size(0)).long()
        if torch.cuda.is_available():
            targets = targets.to('cuda')

        sim_scores = F.log_softmax(sim_scores, dim=1)
        _, preds = torch.max(sim_scores, dim=1)

        loss = criterion(sim_scores, targets)
        scaler.scale(loss).backward()

        batch_loss += loss.cpu().item()
        batch_acc += torch.sum(preds.cpu() == targets.cpu())

    return batch_loss, batch_acc


def evaluating_per_step(training_args, model_args, batch, p_encoder, q_encoder):
    batch_acc = 0

    if torch.cuda.is_available():
        batch = tuple(t.cuda() for t in batch)

    if 'roberta' in model_args.retrieval_model_name_or_path:
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
    targets = torch.arange(0, sim_scores.size(0)).long()

    batch_acc += torch.sum(preds.cpu() == targets)

    return batch_acc


def train_retrieval(training_args, model_args, data_args, tokenizer, p_encoder, q_encoder):

    train_dataloader, eval_dataloader = get_dataloader(tokenizer, training_args, model_args, data_args)

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
        lr=training_args.retrieval_learning_rate,
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
            batch_loss, batch_acc = training_per_step(training_args, model_args, batch,
                                                      p_encoder, q_encoder, criterion, scaler)
            running_loss += batch_loss / training_args.per_device_retrieval_train_batch_size
            running_acc += batch_acc / training_args.per_device_retrieval_train_batch_size
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
                batch_acc = evaluating_per_step(training_args, model_args, batch, p_encoder, q_encoder)

                running_acc += batch_acc / training_args.per_device_retrieval_eval_batch_size
                num_cnt += 1

        eval_epoch_acc = float((running_acc / num_cnt) * 100)
        print(f'Epoch-{epoch} | Accuracy: {eval_epoch_acc:.2f}')

        if eval_epoch_acc > best_acc:
            best_epoch = epoch
            best_acc = eval_epoch_acc

            p_save_path = os.path.join(training_args.retrieval_output_dir, 'p_encoder')
            q_save_path = os.path.join(training_args.retrieval_output_dir, 'q_encoder')

            p_encoder.encoder.save_pretrained(p_save_path)
            q_encoder.encoder.save_pretrained(q_save_path)
            print(f'\t===> best model saved - {best_epoch} / Accuracy: {best_acc:.2f}')

        wandb.log({
            'train/loss': train_epoch_loss,
            'train/learning_rate': scheduler.get_last_lr()[0],
            'eval/epoch_acc': eval_epoch_acc,
            'epoch': epoch,
        })

    return best_acc


if __name__ == '__main__':
    main()
