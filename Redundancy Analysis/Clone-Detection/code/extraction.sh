#!/bin/bash
#request SLURM resources
module load ml-gpu

TRAIN_DATA="$PROJECT_ROOT/dataset/train.txt"
EVAL_DATA="$PROJECT_ROOT/dataset/valid.txt"
TEST_DATA="$PROJECT_ROOT/dataset/test.txt"


DATA=$1

if [ "$DATA" = "train" ]; then

          EXTRACT_DATA="$TRAIN_DATA"

elif [ "$DATA" = "dev" ]; then

          EXTRACT_DATA="$EVAL_DATA"

elif [ "$DATA" = "test" ]; then

          EXTRACT_DATA="$TEST_DATA"
fi

TASK=$2

if [ "$TASK" = "finetune" ]; then

          DO="do_train --do_eval --do_test"

elif [ "$TASK" = "test" ]; then
	  DO="do_test"

elif [ "$TASK" = "extract" ]; then

          DO="do_extract"

fi

MODEL_SETTINGS=$3
if [ "$MODEL_SETTINGS" = "CodeBERT" ]; then

          MODEL="microsoft/codebert-base"
	  MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/CodeBERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "GraphCodeBERT" ]; then

          MODEL="microsoft/graphcodebert-base"
	  MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/GraphCodeBERT"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "UniXCoder" ]; then

          MODEL="microsoft/unixcoder-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/UniXCoder"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "RoBERTa" ]; then

          MODEL="roberta-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/RoBERTa"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "BERT" ]; then

          MODEL="bert-base-uncased"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./saved_models/BERT"
          TOKENIZER=$MODEL


elif [ "$MODEL_SETTINGS" = "CodeGPT-java" ]; then

          MODEL="microsoft/CodeGPT-small-java"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/java-original"
          TOKENIZER=$MODEL

elif [ "$MODEL_SETTINGS" = "CodeGPT-python" ]; then

          MODEL="microsoft/CodeGPT-small-py"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/python-original"
          TOKENIZER=$MODEL



fi


cd $PROJECT_ROOT/code/CodeBERT

# Run Python script with dynamically configured parameters
CUDA_LAUNCH_BLOCKING=1 ml-gpu python run.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=$MODEL_TYPE \
    --output_file=$OUTPUT_DIR/${DATA}_activations.json \
    --config_name=$MODEL \
    --model_name_or_path=$MODEL \
    --tokenizer_name=$TOKENIZER \
    --train_data_file=$TRAIN_DATA \
    --eval_data_file=$EVAL_DATA \
    --test_data_file=$TEST_DATA \
    --$DO \
    --extract_data_file=$EXTRACT_DATA \
    --epoch 2 \
    --block_size 400 \
    --train_batch_size 3 \
    --eval_batch_size 4 \
    --learning_rate 5e-5 \
    --evaluate_during_training \
    --max_grad_norm 1.0 \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11,12 \
    --sentence_only --seed 123456 2>&1| tee $OUTPUT_DIR/${DATA}_${TASK}.log
