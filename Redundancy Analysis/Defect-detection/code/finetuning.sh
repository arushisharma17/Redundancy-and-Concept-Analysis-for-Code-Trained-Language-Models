#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=128G
#SBATCH --time=10-10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-pcie:1
module load ml-gpu

# Change the directory path to a generic placeholder or use an environment variable
cd $PROJECT_ROOT/Codebert

# Ensure the Python environment path is set via an environment variable
$ML_GPU_ENV/python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --config_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --tokenizer_name=microsoft/codebert-base \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file=$PROJECT_ROOT/dataset/train.jsonl \
    --eval_data_file=$PROJECT_ROOT/dataset/valid.jsonl \
    --test_data_file=$PROJECT_ROOT/dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee train.log
