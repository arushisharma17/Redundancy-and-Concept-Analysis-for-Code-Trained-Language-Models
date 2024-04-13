#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=arushi17@iastate.edu  # email address
#SBATCH --mail-type=END
module load ml-gpu


MODEL_SETTINGS=$1
if [ "$MODEL_SETTINGS" = "CodeBERT" ]; then

          MODEL="microsoft/codebert-base"
	  MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/CodeBERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "GraphCodeBERT" ]; then

          MODEL="microsoft/graphcodebert-base"
	  MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/GraphCodeBERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeBERTa" ]; then

          MODEL="huggingface/CodeBERTa-small-v1"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/CodeBERTa"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "UniXCoder" ]; then

          MODEL="microsoft/unixcoder-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/UniXCoder"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "RoBERTa" ]; then

          MODEL="roberta-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./code/saved_models/RoBERTa"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "BERT" ]; then

          MODEL="bert-base-uncased"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./code/saved_models/BERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "JavaBERT" ]; then

          MODEL="CAUKiel/JavaBERT"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./code/saved_models/JavaBERT"
          #TOKENIZER="bert-base-cased"
	  TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "GPT2" ]; then

          MODEL="gpt2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/GPT2"
          TOKENIZER=$MODEL #need to add padding token 
elif [ "$MODEL_SETTINGS" = "CodeGPT-java" ]; then

          MODEL="microsoft/CodeGPT-small-java"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/CodeGPT/java-original"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-python" ]; then

          MODEL="microsoft/CodeGPT-small-py"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/CodeGPT/python-original"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-java-adapted" ]; then

          MODEL="microsoft/CodeGPT-small-java-adaptedGPT2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/CodeGPT/java-adapted"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-python-adapted" ]; then

          MODEL="microsoft/CodeGPT-small-py-adaptedGPT2"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./code/saved_models/CodeGPT/python-adapted"
          TOKENIZER=$MODEL

fi



cd /work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/NL-Code-Search/NL-code-search-WebQuery
ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/python code/run_classifier.py \
                        --model_type $MODEL_TYPE \
                        --do_train \
                        --do_eval \
                        --eval_all_checkpoints \
                        --train_file train_codesearchnet_7.json \
                        --dev_file dev_codesearchnet.json \
                        --max_seq_length 200 \
                        --per_gpu_train_batch_size 8 \
                        --per_gpu_eval_batch_size 8 \
                        --learning_rate 1e-5 \
                        --num_train_epochs 3 \
                        --gradient_accumulation_steps 1 \
                        --warmup_steps 1000 \
                        --evaluate_during_training \
                        --data_dir ./data/ \
                        --output_dir ${OUTPUT_DIR}/model_codesearchnet \
                        --encoder_name_or_path $MODEL | tee ${OUTPUT_DIR}/train.log


ml-gpu /work/LAS/jannesar-lab/arushi/Environments/finetuning_env/bin/python code/run_classifier.py \
                        --model_type $MODEL_TYPE \
                        --config_name $MODEL \
                        --tokenizer_name $MODEL \
                        --do_train \
                        --do_eval \
                        --eval_all_checkpoints \
                        --train_file cosqa-train.json \
                        --dev_file cosqa-dev.json \
                        --max_seq_length 200 \
                        --per_gpu_train_batch_size 8\
                        --per_gpu_eval_batch_size 8 \
                        --learning_rate 1e-5 \
                        --num_train_epochs 3 \
                        --gradient_accumulation_steps 1 \
                        --warmup_steps 5000 \
                        --evaluate_during_training \
                        --data_dir ./data/ \
                        --output_dir ${OUTPUT_DIR}/model_cosqa_continue_training \
                        --encoder_name_or_path ${OUTPUT_DIR}/model_codesearchnet/checkpoint-best-aver | tee ${OUTPUT_DIR}/continue_train.log





