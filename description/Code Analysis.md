# Code Analysis
코드가 실행되는 순서에 따라 전반적인 설명을 진행한다.

<br>

### 메인 함수 실행
> train.py 368~369

```py
if __name__ == "__main__":
    main()
```

### Argument 및 model loading
> train.py main 39~42

```py
parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```
* `HfArgumentParser`는 기존의 `Argparse`의 subset이다. 사용자가 만든 Class또는 존재하는 Class에서 선언한 인자들을 추후에 사용하기 위함이며 command line으로 입력한 변수로 내부 값을 변경 가능하다.
* `ModelArguments`는 모델을, `DataTrainingArguments`는 토크나이저를, `TrainingArguments`는 학습을 위한 설정을 위한 인자들을 미리 선언해놓은 클래스이다.
* transformer에서 제공하는 `HfArgumentParser`와 `TrainingArguments`를 사용했고 arguments.py에서 `ModelArguments`와 `DataTrainingArguments`를 정의한다.
* `parser.parse_args_into_dataclasses()`를 사용해 사전에 정의된 인자를 세 개의 변수에 할당할 수 있다. 이 변수를 이용해 Auto series(ex AutoConfig, AutoModel, etc)를 가지고 각 항목을 불러온다.

`ModelArguments`와 `DataTrainingArguments`를 알아보자.

> arguments.py ModelArguments 11~28

```py
model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
```
* load할 model과 config, tokenizer를 정의할 수 있다.

> arguments.py DataTrainingArguments 37~92

```py
	dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=1,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
```
* load한 tokenizer에 대한 세부 인자를 설정할 수 있다.

> train.py main 53~63

```py
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)
```
* log를 설정하고 seed를 고정한다.

> train.py main 65~100

```py
    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
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
        config=config,
    )
    
    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)
```
* `load_from_disk`함수로 dataset을 불러오며, Dataset과 DatasetDict의 두 가지 경우로 dataset을 반환받는데, 여기서는 DatasetDict로 받아지며 이는 dataset['train']과 dataset['validation']으로 나눠질 수 있게한다.
* 이전에 `parser.parse_args_into_dataclasses()`로 얻은 세 변수로 모델과 Config, 토크나이저를 불러온다.
* `main` 함수는 마지막 두 줄을 끝으로 종료된다. 만약 `do_train` 또는 `do_eval`이 True일 경우에만 `run_mrc` 함수가 실행되며, 이 두 값은 default가 False이다. 따라서 train.py 실행 시 꼭 이 두 변수 중 한 변수를 True로 설정해줘야한다.
* `run_mrc`는 3개의 arg 변수와 이 변수들로 불러온 datasets, tokenizer, model을 인자로 입력한다.


### MRC
> train.py run_mrc 103~130

```py
def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
```
* `run_mrc` 함수가 정의된다. `do_train`에 따라 train 또는 validation의 데이터셋의 column이 선택된다.
* 기본적으로 question, context, answer에 대한 컬럼명을 이름 그대로 지으며, 만약 데이터셋에서 따로 사용하는 컬럼명이 있다면 그것을 따른다.
* padding을 오른쪽에 추가한다.
* argument와 dataset, tokenizer에 오류가 있는지 확인하며 이는 `check_no_error` 함수로 검사한다. 오류가 존재하면 에러가 발생되며 존재하지 않으면 튜플 형태로 `checkpoint`와 `max_seq_length`가 반환된다.

> utils_qa.py check_no_error 319~343

```py
def check_no_error(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
) -> Tuple[Any, int]:

    # last checkpoint 찾기.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
```
* 우선 last_checkpoint를 찾는것이 목적이며 이를 위해서 다음 3가지를 검사한다.
  * `output_dir`이 존재하는가? (이는 우리가 직접 인자로 정해주어야 한다)
  * `valid`가 아니라 `train`을 위한 목적인가?
  * `output_dir`에 있는 기존 모델을 사용할 것인가? 사용할 것이라면 `overwrite_output_dir`이 False이며, 새로 학습을 할 것이라면 True가 된다.
* `output_dir`에서 checkpoint를 불러온다. `get_last_checkpoint`는 transformers.trainer_utils 라이브러리에 있는 함수로 가장 마지막에 생성된 체크포인트를 가져온다.
* 이 때, 체크포인트가 있으면 불러오고, 없으면 없는대로 학습을 시작하지만, 체크포인트가 없으면서 `output_dir`이 빈 폴더가 아니라면 `overwrite_output_dir`가 False인데 덮어쓸 우려가 있으므로 에러를 발생시킨다.


> utils_qa.py check_no_error 345~362

```py
    # Tokenizer check: 해당 script는 Fast tokenizer를 필요로합니다.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    return last_checkpoint, max_seq_length
```
* `isinstance(A, B)`는 A가 Class B의 인스턴스인지를 검사하는 함수이며, tansformers의 AutoTokenizer.from_pretrained로 로딩되는 모든 토크나이저(단 Fast 버전으로 로딩되었다고 가정한다)는 PreTrainedTokenizerFast를 상속받기 때문에 그렇지 않으면 에러가 발생된다. 위쪽에서 tokenizer를 로드할 때 `use_fast=True`로 설정했다.
* `data_args.max_seq_length`는 토크나이징을 거친 최대 input sequence 길이를 DataTrainingArguments에서 정의해준 것으로 모델의 max_seq_length는 언제든지 변경가능하고 data의 arg보다 커도 문제가 없지만 data의 max_seq_length가 더 크면 문제가 되므로 이에 대한 예외처리를 해준다.
* 검증을 위한 데이터셋이 존재하지 않으면 에러를 발생시킨다.

> train.py run_mrc 210~222

```py
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
```
* 학습을 위해 `train_dataset`을 선언하며 `map`을 사용하면 dataset에 대한 처리들을 한번에 할 수 있다. `map`은 python 내장 함수가 아니라 `dataset`클래스에서 제공하는 함수이며 인자로 받은 함수들을 순서대로 들어가게된다.
* 이 때 map을 위한 몇 개의 함수들이 사전에 정의되어있으며 `prepare_train_feautures`를 제외하고는 모두 이에 해당한다.
  * `batched=True`는 말 그대로 데이터셋을 배치화하며 `batch_size` 인자로 크기를 지정해줄 수 있다. default는 1000이다.
  * `num_proc`으로 Multiprocessing을 설정한다.
  * `remove_columns`로 인자로 받은 컬럼들을 쉽게 drop할 수 있다.
  * 캐시 파일은 매 학습시마다 생성되며 세션이 끝나면 삭제된다. `load_from_cahce_file=False`라면 데이터셋을 캐시에서 불러오지 않고 디스크에서 불러오게 된다. 캐싱의 장점은 학습 시 데이터셋을 매번 디스크에서 가져와야 되는 작업을 미리 캐시에 담아놓는 다는 점에서 학습을 빠르게 할 수 있도록 한다.

dataset.map으로 처음 적용되는 `prepare_train_features`에 대해 알아보자
> train.py prepare_train_features 132~208

```py
    # Train preprocessing / 전처리를 진행합니다.
    def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            #return_token_type_ids=False, # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
```
* 학습을 위해 train dataset을 전처리하는 과정이다.

###
> 
```py

```

###
> 
```py

```
###
> 
```py

```
