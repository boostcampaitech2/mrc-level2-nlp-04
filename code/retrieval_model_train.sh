#python retrieval_train.py \
#--project_name dense_retrieval_implement \
#--retrieval_run_name roberta-small \
#--use_trained_model False \
#--retrieval_model_name_or_path klue/roberta-small
#
#python retrieval_train.py \
#--project_name dense_retrieval_implement \
#--retrieval_run_name roberta-base \
#--use_trained_model False \
#--retrieval_model_name_or_path klue/roberta-base
#
#python retrieval_train.py \
#--project_name dense_retrieval_implement \
#--retrieval_run_name roberta-large \
#--use_trained_model False \
#--retrieval_model_name_or_path klue/roberta-large \
#--per_device_retrieval_train_batch_size 4 \
#--per_device_retrieval_eval_batch_size 4
#
#python retrieval_train.py \
#--project_name dense_retrieval_implement \
#--retrieval_run_name bert-base \
#--use_trained_model False \
#--retrieval_model_name_or_path klue/bert-base

python retrieval_train.py \
--project_name dense_retrieval_implement \
--retrieval_run_name koelectra \
--use_trained_model False \
--retrieval_model_name_or_path monologg/koelectra-base-v3-discriminator \
--per_device_retrieval_train_batch_size 4 \
--per_device_retrieval_eval_batch_size 4