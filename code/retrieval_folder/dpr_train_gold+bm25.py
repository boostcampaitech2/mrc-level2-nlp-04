from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import wandb
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from bm25_first_passage_from_wiki import get_bm25_passage


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


set_seed(42)  # magic number :)

wandb.init(project='DPR_gold_test', entity='ssp', name='roberta-small_DPR_batch_8_epoch_5')

try:
    dataset = load_from_disk("mrc-level2-nlp-04/data/train_dataset")
except:
    dataset = load_from_disk("../../data/train_dataset")

model_checkpoint = "klue/roberta-small"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
configuration = AutoConfig.from_pretrained(model_checkpoint)
emb_size = configuration.max_position_embeddings
training_dataset = dataset['train'][:2000]

q_seqs = tokenizer(
    training_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
p_seqs = tokenizer(
    training_dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
answers = training_dataset['answers']

ans_seqs = []
for answer in answers:
    ans_seqs.append(answer["text"][0])

bm25_passages = get_bm25_passage(training_dataset["question"], ans_seqs)

p_bm25_seqs = tokenizer(bm25_passages, padding="max_length",
                        truncation=True, return_tensors='pt')

train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
                              p_bm25_seqs['input_ids'], p_bm25_seqs['attention_mask'], p_bm25_seqs['token_type_ids'],
                              q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'],
                              )


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)
        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

        pooled_output = outputs[1]

        return pooled_output


# load pre-trained model on cuda (if available)
p_encoder = BertEncoder.from_pretrained(model_checkpoint)
q_encoder = BertEncoder.from_pretrained(model_checkpoint)

if torch.cuda.is_available():
    p_encoder.cuda()
    q_encoder.cuda()


def train(args, dataset, p_model, q_model):

    # Dataloader
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(
        dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in p_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in p_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in q_model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in q_model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    t_total = len(
        train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    # Start training!
    global_step = 0

    p_model.zero_grad()
    q_model.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            q_encoder.train()
            p_encoder.train()


            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            # (2*batch_size, emb_size)
            p_inputs = {'input_ids': torch.cat((batch[0], batch[3]), dim=0),
                        'attention_mask': torch.cat((batch[1], batch[4]), dim=0),
                        'token_type_ids': torch.cat((batch[2], batch[5]), dim=0),
                        }
            # (batch_size, emb_size)
            q_inputs = {'input_ids': batch[6],
                        'attention_mask': batch[7],
                        'token_type_ids': batch[8]}

            p_outputs = p_model(**p_inputs)  # (2*batch_size, hidden_dim)
            q_outputs = q_model(**q_inputs)  # (batch_size, hidden_dim)


            # Calculate similarity score & loss
            # (batch_size, emb_dim) x (emb_dim, 2*batch_size) = (batch_size, 2*batch_size)
            sim_scores = torch.matmul(
                q_outputs, torch.transpose(p_outputs, 0, 1))

            # target: position of positive samples = diagonal element
            targets = torch.arange(0, 2 * args.per_device_train_batch_size, 2).long()
            if torch.cuda.is_available():
                targets = targets.to('cuda')

            sim_scores = F.log_softmax(sim_scores, dim=1)

            loss = F.nll_loss(sim_scores, targets)
            wandb.log({'Train loss': loss})
            if step % 10 == 0:
                print(f"step: {step}, loss: {loss.item()}")

            loss.backward()
            optimizer.step()
            scheduler.step()
            q_model.zero_grad()
            p_model.zero_grad()
            global_step += 1

            torch.cuda.empty_cache()

    return p_model, q_model


args = TrainingArguments(
    output_dir="dense_retrieval",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01
)

p_encoder, q_encoder = train(args, train_dataset, p_encoder, q_encoder)
