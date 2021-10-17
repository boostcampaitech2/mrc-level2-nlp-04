import logging
import os
import sys

import torch
import wandb
from datasets import load_metric
from torch.cuda.amp import autocast
from tqdm import trange, tqdm

from utils_qa import AverageMeter, post_processing_function, create_and_fill_np_array, get_args, set_seed_everything, \
    get_models, get_data, get_optimizers

logger = logging.getLogger(__name__)

# avoid huggingface/tokenizers: The current process just got forked, after parallelism has already been used.
# Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def validation_per_steps(model, datasets, valid_loader, valid_dataset, training_args, data_args):
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


def train_mrc(model, optimizer, scaler, scheduler, datasets, train_loader, valid_loader, valid_dataset,
              training_args, model_args, data_args):
    """
    train & validation 함수
    """
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
                    valid_metrics = validation_per_steps(model, datasets, valid_loader, valid_dataset,
                                                         training_args, data_args)
                if valid_metrics['exact_match'] > prev_em:
                    torch.save(model, os.path.join(training_args.output_dir,
                                                   f'{model_args.model_name_or_path.split("/")[-1]}.pt'))
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

    if project_name is not None:
        training_args.project_name = project_name
    if model_name_or_path is not None:
        model_args.model_name_or_path = model_name_or_path

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed_everything(training_args.seed)

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.project_name,
                                            model_args.model_name_or_path.split('/')[-1])
    training_args.per_device_train_batch_size = 16
    training_args.per_device_eval_batch_size = 128
    training_args.num_train_epochs = 10
    training_args.learning_rate = 3e-5
    training_args.warmup_steps = 1000
    training_args.weight_decay = 0.01
    training_args.logging_steps = 100
    training_args.fp16 = True
    print(training_args)
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    tokenizer, model_config, model = get_models(training_args, model_args)
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

    train_mrc(model, optimizer, scaler, scheduler, datasets, train_loader, valid_loader, valid_dataset,
              training_args, model_args, data_args)
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
