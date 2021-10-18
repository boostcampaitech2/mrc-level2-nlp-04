# Code Analysis
코드가 실행되는 순서에 따라 전반적인 설명을 진행한다.

```
┌ Train
│　　├─── 메인 함수 실행
│　　├─── Argument 및 model loading
│　　├─── MRC
│　　├─── Preprocessing
│　　├─── Data Collator
│　　├─── Metric
│　　├─── Train (작성 중)
│　　├─── Postprocessing (작성 중)
├ Retrieval
│　　├─── (작성 예정)
└ Inference
　 　├─── (작성 예정)
```

---

<br>

## Train

### 메인 함수 실행
> train.py 368~369

```py
if __name__ == "__main__":
    main()
```

---

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

---

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

---

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
  * overwrite_cache : 캐싱을 이용해 더 빨리 데이터를 배치만큼 가져오도록 할 지
  * eval_retrieval : retrieval 할 때 sparse embedding 방식을 사용할 것인지
  * num_clusters : 비슷한 passage끼리 모아놓은 군집을 몇개로 설정할 지
  * tok_k_retrieval : 상위 몇개의 passage를 retrieve 할지
  * use_faiss : passage retrival을 faiss를 사용해서 할지
---

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

---

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

---

<br>

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

---

위에서 사용된 `check_no_error` 함수에 대해 알아보자.

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

---

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

---

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

---

<br>

### Preprocessing

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
* 학습을 위해 train dataset을 전처리하는 과정이다. 단방향 모델에서는 역방향으로 시퀀스에 접근할 수 있기 때문에 pad_on_right의 경우까지 고려해준다.
* tokenizer를 통해 data를 tokenize한다. 이 때의 truncation은 context에 대해서만 truncate한다. question의 길이가 그만큼 길지 않기 때문이기도 하다.
* overflowing_tokens와 offsets_mapping에 대한 값을 각각 변수로 저장한다. 이후, start와 end position이 담길 리스트를 생성하고 이를 dictionary로 매핑한다.
* [for] 반복문을 돌며, 이 때 max_length로 truncation된 시퀀스별로 접근하게 된다.
  * offsets은 `[(0, 0), (0, 1), (1, 2), (2, 3), (0, 0)]` 꼴의 데이터가 매번 담긴다.
  * 각 토큰들의 input_ids와 cls를 기억한다.
  * `sequence_ids`는 None, 0, 1의 값을 가지며 앞쪽 시퀀스는 0, 뒤쪽 시퀀스는 1을 반환하게 된다. None은 스페셜 토큰들이 반환하는 값이다. 이 값들을 변수로 저장한다.
  * sample_mapping에 해당하는 context의 answer를 가져온다.
  * [if] answer가 없는 경우는 현재 train_dataset에는 존재하지 않는다. 이는, retrieval로 가져온 data나 augmentation을 거친 data가 answer가 없을 수 있어서 존재하는 조건문인 듯 보인다. 이때 답이 없음을 가리키는 cls_index를 추가한다.
  * [else] 만약 answer가 있을 때 `[{'answer_start': [235], 'text': ['하원']}`와 같은 꼴로 존재하며, 이 두 정보를 이용해 start_char와 end_char를 기억한다. 이 예에서는 start_char=235, end_char=237이 된다.
  * [else] 두 개의 while문을 통해 sequence_ids가 1이 되는 지점을 찾는다. 현재 sequence_ids가 1이라는 뜻은 context에 해당하는 token을 찾겠다는 뜻이며 이는 context의 시작 위치와 끝 위치를 찾는 과정이다.
  * [else-if] 정답의 시작(=start_char)은 무조건 start_index보다는 뒤에 있어야 하고 정답의 끝(=end_char)은 무조건 end_index보다는 앞에 있어야 한다. 그렇지 않으면(=not) cls_index를 추가한다.
  * [else-else] 정답이 text의 span안에 있다면, text를 점점 answer에 가깝게 span을 줄여나간다. 제일 가깝게 줄였을 때 start_index와 end_index를 답으로 추가한다.
  
---

validation dataset에 대해서도 `do_eval==True`의 경우를 위해 정의해준다. 방식은 train dataset과 동일하다.

> train.py run_mrc 263~273

```py
    if training_args.do_eval:
        eval_dataset = datasets["validation"]

        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
```

---

전처리 과정에서는 tokenizing하는 구성은 동일하나 데이터셋에 대한 따로 answer의 유무와 start_position, end_position에 대한 처리가 따로 없이 context에 대한 token들만 반환된다.

> train.py prepare_validation_features 224~261

```py
    # Validation preprocessing
    def prepare_validation_features(examples):
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

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        return tokenized_examples
```

---

<br>

### Data Collator
현재는 패딩을 max_seq_length에 맞춰서 하고 있지만 이를 기준으로 하지 않고, 각 배치의 max_length를 기준으로 패딩을 하게되면 좀 더 효율적으로 padding을 진행할 수 있다.

> train.py run_mrc 278~280

```py
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )
```

---

<br>

### Metric
> train.py run_mrc 308~311

```py
    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
```
* EM과 F1위주의 평가 지표를 사용할 것이므로 `squad`의 metric을 로딩한다.
* `metric.compute`는 torch에서 제공하는 함수로 pred와 label을 정해진 metrics를 가지고 계산한 값을 반환한다.
* 이 함수는 train하는 과정에서 실행되기 때문에 순서에 맞게 소개된 부분은 아니지만, 그 순서가 바로 다음이라는 점, metric을 load하는 코드와 같이 설명하는 것이 좋겠다는 점에서 같이 이야기 되었다.

---

<br>

### Train

> train.py run_mrc 313~324

```py
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
    )
```
* trainer는 기존 huggingface에서 제공하는 Trainer를 사용하되, 이를 상속받아 좀 더 개선해서 QA Task에 specific한 trainer를 사용한다.
* 여기서 사용하는 `data_collator`와 `compute_metrices`는 위에서 언급한 함수이며, `post_processing_function`은 아래에서 다시 다룬다.

---

> train_qa.py QuestionAnsweringTrainer 30~35

```py
# Huggingface의 Trainer를 상속받아 QuestionAnswering을 위한 Trainer를 생성합니다.
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
```
* 우리가 사용하는 Trainer이며, 실제 허킹페이스 깃허브에도 업로드 되어있다. train시 사용될 인자 `args`와 `kwargs`를 입력받아 `super().__init__()`에서 사용하며, 몇몇 함수에서 사용할 인자와 후처리 함수를 입력으로 받는다.
* 여기에는 validation dataset에서 사용하는 `evaluate` 함수와 test dataset에서 사용하는 `predict` 함수가 있으며, 이는 나중에 호출될 때 다시 다룬다.

---


> train.py run_mrc 326~355

```py
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
```
* chkecpoint가 있으면 불러오며, 학습을 진행한다. 학습이 끝나면 모델을 저장한다.
* metric에 전체 데이터셋 길이를 알려주고 계산된 metric이 logging되고 save될 수 있도록 한다. trainer의 파라미터도 저장한다.
* 그 외에도 결과 파일들을 저장한다.

---

> train.py run_mrc 357~365

```py
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
```
* 평가 모드라면 학습을 하지 않고 단순히 `evaluate`로 평가만 진행하며 이에 대한 metrics을 저장한다.

---

<br>

### Postprocessing


> train.py post_processing_function 283~306

```py
    def post_processing_function(examples, features, predictions, training_args):
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

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )
```
* 트레이너에 입력되는 후처리 함수이다. 이에 대한 자세한 인자 설명은 `postporcess_qa_predictions`의 주석으로 달려있다. 이 함수를 통해 얻은 predictions를 id와 text의 key를 가진 딕셔너리 형태로 만든다.
* 학습으로 실행된 것이면 id와 text로 이루어진 포맷을 반환하고, 검증으로 실행된 것이면 validation dataset의 라벨과 포맷을 함께 EvalPrediction Class로 넘겨준다. 이는 나중에 `p.predictions`, `p.label_ids` 와 같은 꼴로 사용할 수 있다. 이는 test는 정답을 모르기에 예측만을 가지고 사용하고 valid는 정답과 비교해 metrics를 출력하기 위함이다.
* 중요한 것은 이 함수는 train이 아닌, valid와 test dataset만이 적용된다는 것이다. 이러한 이유는, train은 후처리를 통해 평가 지표를 비교하지 않고, start pos와 end pos에 대한 loss로 update되기 때문이다. 실제로 train.py에서는 training_args.do_predict를 사용할 일이 없으며 이에 대해서는 inference와 동일한 코드 사용을 위해 추가되어 있는 것으로 보인다. (그래서, train.py보다는 utils_qa쪽에 있는 것이 더 자연스럽고 재사용성이 좋다. 현재는 inference.py에도 동일한 코드가 추가되어있기 때문)

---

위 코드에서 postprocessing을 하기 전에 이루어진 예측이 어떻게 이루어지는지 확인해본다. 여기서는 8개의 구역으로 나누어 250줄의 코드를 설명한다.

> utils_qa.py postprocess_qa_predictions 60~102

```py
def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes : qa model의 prediction 값을 후처리하는 함수
    모델은 start logit과 end logit을 반환하기 때문에, 이를 기반으로 original text로 변경하는 후처리가 필요함

    Args:
        examples: 전처리 되지 않은 데이터셋 (see the main script for more information).
        features: 전처리가 진행된 데이터셋 (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            모델의 예측값 :start logits과 the end logits을 나타내는 two arrays              첫번째 차원은 :obj:`features`의 element와 갯수가 맞아야함.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            정답이 없는 데이터셋이 포함되어있는지 여부를 나타냄
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            답변을 찾을 때 생성할 n-best prediction 총 개수
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            생성할 수 있는 답변의 최대 길이
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            null 답변을 선택하는 데 사용되는 threshold
            : if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            아래의 값이 저장되는 경로
            dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            dictionary에 `prefix`가 포함되어 저장됨
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            이 프로세스가 main process인지 여부(logging/save를 수행해야 하는지 여부를 결정하는 데 사용됨)
    """
```
* `postprocess_qa_predictions` 함수는 위와 같은 인자를 받으며 이에 대한 설명은 주석으로 자세히 명시되어있다.

---

> utils_qa.py postprocess_qa_predictions 103~128

```py
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # example과 mapping되는 feature 생성
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # prediction, nbest에 해당하는 OrderedDict 생성합니다.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )
```
* prediction은 (start_pos, end_pos)라는 튜플형으로 존재해야 하며 이에 대한 잘못된 포맷사용을 막아준다. 
* 또, 전체 예측의 개수와 실제 데이터의 개수가 동일해야 한다.
* `example_id_to_index`는 단순히 각각의 example을 0부터 indexing하기 위한 변수이다. `example_id`는 mrc-1-000067와 같은 값을 가지며 이들에 대한 순서를 메기기 위해 사용한다.
* `features_per_example`은 example과 feature를 mapping하기 위한 함수이다. example과 feature는 개수가 다르고 순서가 다를 수 있기 때문에 이를 이어준다. 왜냐하면 feature는 example에서 max_seq_length로 split(=truncate)되었기 때문.
* 순서를 고려하는 OrderedDict를 생성합니다. all_predictions는

---

> utils_qa.py postprocess_qa_predictions 130~168

```py
    # 전체 example들에 대한 main Loop
    for example_index, example in enumerate(tqdm(examples)):
        # 해당하는 현재 example index
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # 현재 example에 대한 모든 feature 생성합니다.
        for feature_index in feature_indices:
            # 각 featureure에 대한 모든 prediction을 가져옵니다.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # logit과 original context의 logit을 mapping합니다.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional : `token_is_max_context`, 제공되는 경우 현재 기능에서 사용할 수 있는 max context가 없는 answer를 제거합니다
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            # minimum null prediction을 업데이트 합니다.
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # `n_best_size`보다 큰 start and end logits을 살펴봅니다.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()

            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
```
* 전체 example에 대한 loop를 돌며 features에 접근한다.
* 각 feature에 대한 loop를 돌며 feature(=truncated context)의 (answer이라고 생각되는)start와 end의 점수를 계속 쌓는다. 계속 점수들이 쌓일텐데, 이 중에 n_best_size만큼만을 선택해서 `tolsit()` 한 최종 결과로 `start_indexes`와 `end_indexes`를 가지게된다. 

---

> utils_qa.py postprocess_qa_predictions 170~202

```py
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # out-of-scope answers는 고려하지 않습니다.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # 길이가 < 0 또는 > max_answer_length인 answer도 고려하지 않습니다.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # 최대 context가 없는 answer도 고려하지 않습니다.
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
```
* 

---

> utils_qa.py postprocess_qa_predictions 204~242

```py
        if version_2_with_negative:
            # minimum null prediction을 추가합니다.
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # 가장 좋은 `n_best_size` predictions만 유지합니다.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # 낮은 점수로 인해 제거된 경우 minimum null prediction을 다시 추가합니다.
        if version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        # offset을 사용하여 original context에서 answer text를 수집합니다.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):

            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # 모든 점수의 소프트맥스를 계산합니다(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # 예측값에 확률을 포함합니다.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob
```

---

> utils_qa.py postprocess_qa_predictions 244~264

```py
        # best prediction을 선택합니다.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # else case : 먼저 비어 있지 않은 최상의 예측을 찾아야 합니다
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # threshold를 사용해서 null prediction을 비교합니다.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable 가능
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]
```

---

> utils_qa.py postprocess_qa_predictions 266~277

```py
        # np.float를 다시 float로 casting -> `predictions`은 JSON-serializable 가능
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]
```

---

> utils_qa.py postprocess_qa_predictions 279~316

```py
    # output_dir이 있으면 모든 dicts를 저장합니다.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"predictions_{prefix}".json,
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"nbest_predictions_{prefix}".json,
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"null_odds_{prefix}".json,
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
            )
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n"
                )

    return all_predictions
```

---

<br>

## Retrieval

<br>

###
> 

```py

```

---

<br>

## Inference

<br>

###
> 

```py

```


<br>
