python retrieval_test.py \
--project_name retrieval_accuracy_comparison \
--retrieval_run_name roberta-small \
--use_trained_model True \
--retrieval_model_name_or_path klue/roberta-small

python retrieval_test.py \
--project_name retrieval_accuracy_comparison \
--retrieval_run_name roberta-base \
--use_trained_model True \
--retrieval_model_name_or_path klue/roberta-base

python retrieval_test.py \
--project_name retrieval_accuracy_comparison \
--retrieval_run_name roberta-large \
--use_trained_model True \
--retrieval_model_name_or_path klue/roberta-large

python retrieval_test.py \
--project_name retrieval_accuracy_comparison \
--retrieval_run_name bert-base \
--use_trained_model True \
--retrieval_model_name_or_path klue/bert-base

python retrieval_test.py \
--project_name retrieval_accuracy_comparison \
--retrieval_run_name koelectra \
--use_trained_model True \
--retrieval_model_name_or_path monologg/koelectra-base-v3-discriminator