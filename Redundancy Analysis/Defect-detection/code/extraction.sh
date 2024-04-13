#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=10-10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-pcie:1
module load ml-gpu

# Use environment variables to define paths
cd $PROJECT_ROOT/Codebert

# Use the environment variable for the Python environment
$ML_GPU_ENV/python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --output_file=train_activations.json \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_extract \
    --train_data_file=$PROJECT_ROOT/dataset/train.jsonl \
    --eval_data_file=$PROJECT_ROOT/dataset/valid.jsonl \
    --test_data_file=$PROJECT_ROOT/dataset/test.jsonl \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --sentence_only --seed 123456 2>&1 | tee train_extraction.log
