#!/bin/bash
### install requirements for pstage3 baseline
# pip requirements
pip install datasets==1.5.0
pip install transformers==4.11.3
pip install tqdm==4.62.3
pip install torch==1.7.1
pip install tokenizers==0.10.3
pip install pandas==1.1.4
pip install scikit-learn==0.24.1
pip install konlpy==0.5.2
pip install numpy==1.19.2
pip install huggingface-hub==0.0.19
pip install elasticsearch==7.10.0
pip install elastic-apm==6.6.0

# faiss install (if you want to)
pip install faiss-gpu
