
## Running Scripts

To run fine-tuning or extraction processes, use the following command:
`sbatch extraction1.sh <dataset-train,dev,test> <task-extract,finetune> <model>`
Example: `sbatch extraction1.sh train extract GraphCodeBERT`. 


All model settings are provided within the script.

## Setting Up Model Directories

First, create a base directory for saved models and then create individual directories for each model:

mkdir saved\_models
cd saved\_models
mkdir CodeBERT GraphCodeBERT UniXCoder RoBERTa BERT CodeGPT-java CodeGPT-python 


### Details of models and tokenizers
1. CodeBERT
          MODEL="microsoft/codebert-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/CodeBERT"
          TOKENIZER=$MODEL

2. GraphCodeBERT

          MODEL="microsoft/graphcodebert-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/GraphCodeBERT"
          TOKENIZER=$MODEL

4. UniXCoder

          MODEL="microsoft/unixcoder-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/UniXCoder"
          TOKENIZER=$MODEL

5. RoBERTa

          MODEL="roberta-base"
          MODEL_TYPE="roberta"
          OUTPUT_DIR="./saved_models/RoBERTa"
          TOKENIZER=$MODEL

6. BERT

          MODEL="bert-base-uncased"
          MODEL_TYPE="bert"
          OUTPUT_DIR="./saved_models/BERT"
          TOKENIZER=$MODEL

9. CodeGPT-java

          MODEL="microsoft/CodeGPT-small-java"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/java-original"
          TOKENIZER=$MODEL

10. CodeGPT-python

          MODEL="microsoft/CodeGPT-small-py"
          MODEL_TYPE="gpt2"
          OUTPUT_DIR="./saved_models/CodeGPT/python-original"
          TOKENIZER=$MODEL


Notes:
While finetuning gpt2 models - used tokenizer.pad_token_id to get sequence length for gpt2 and ada[ted models (the ones pretrained on text) and used model.config.pad_token_id for CodeGPT - original models for java and python. Tha token ids according to tokenizer and model are slightly different for the adapted models causing errors. like 50256, 50257

In case of original models the model config ones are 0,1,2 for bos,eos,pad but 5025smth according to the tokenizer.



