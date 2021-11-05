<div align="center">
  <h1>MRC Open-Domain Question Answering</h1>
</div>

![](code/assets/대회이미지.png)

## :fire: Getting Started

### Data setting & Install Requirements

```
# data (51.2 MB)
tar -xzf data.tar.gz

# 필요한 파이썬 패키지 설치. 
bash ./install/install_requirements.sh
```

### Dependencies

```
datasets==1.5.0
elasticsearch==7.10.0
elastic-apm==6.6.0
huggingface-hub==0.0.19
numpy==1.19.2
pandas==1.1.4
transformers==4.11.3
tqdm==4.62.3
torch==1.7.1
tokenizers==0.10.3
```

### Contents

```bash
./code/assets/                  # readme 에 필요한 이미지 저장
./code/install/                 # 요구사항 설치 파일 
./code/model/                   # Additional custom model 파일
./code/arguments.py             # 실행되는 모든 argument가 dataclass 의 형태로 저장되어있음 
./code/combine.py               # k-fold 적용 후 soft voting 하는 코드 제공
./code/data_processing.py       # reader 모델을 위한 데이터 전처리 모듈 제공
./code/dense_retrieval.py       # dense retriever 모듈 제공
./code/retrieval.py             # sparse retreiver 모듈 제공
./code/elastic_search.py        # elastic search 모듈 제공
./code/prepare_dataset.py       # 데이터셋의 context 를 전처리하고 저장해주는 파일
./code/retrieval_train.py       # dense retrieval model 학습 
./code/retrieval_test.py        # retrieval (sparse, dense, elastic) top-k 에 따른 성능 비교
./code/train.py                 # MRC, Retrieval 모델 학습 및 평가 
./code/inference.py		        # ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성
./code/trainer_qa.py            # MRC 모델 학습에 필요한 trainer 제공.
./code/utils_qa.py              # reader 모델 관련 유틸 함수 제공
./code/utils_retrieval.py       # retrieval 모델 관련 유틸 함수 제공 
./data/                         # 전체 데이터. 아래 상세 설명
./description/                  # baselie code 설명
```

## Data Information

아래는 제공하는 데이터셋의 분포를 보여줍니다.

![](code/assets/데이터구성.png)

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

### ./data structure

```python
./data/                        # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성 
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다. 

## :mag: Overview

### Background

Question Answering (QA)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다. 
다양한 QA 시스템 중, Open-Domain Question Answering (ODQA) 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 
Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가되기 때문에 더 어려운 문제입니다.

![img.png](img.png)

본 ODQA 대회에서 우리가 만들 모델은 two-stage로 구성되어 있습니다. 첫 단계는 질문에 관련된 문서를 찾아주는 "retriever" 단계이고, 
다음으로는 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 "reader" 단계입니다. 두 가지 단계를 각각 구성하고 그것들을 적절히 통합하게 되면, 
어려운 질문을 던져도 답변을 해주는 ODQA 시스템을 여러분들 손으로 직접 만들어보게 됩니다.

따라서, 대회는 더 정확한 답변을 내주는 모델을 만드는 팀이 좋은 성적을 거두게 됩니다.
