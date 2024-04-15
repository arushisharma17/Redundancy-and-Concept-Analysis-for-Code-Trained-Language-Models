# Clone Detection Task

## Setting Up Model Directories

First, create a base directory for saved models and then create individual directories for each model:

mkdir saved\_models
cd saved\_models
mkdir CodeBERT GraphCodeBERT UniXCoder RoBERTa BERT CodeGPT-java CodeGPT-python 

## Running Scripts

To run fine-tuning or extraction processes, use the following command:
`sbatch extraction1.sh <dataset-train,dev,test> <task-extract,finetune> <model>`
Example: `sbatch extraction1.sh train extract GraphCodeBERT`. 


All model settings are provided within the script.





Notes:
While finetuning gpt2 models - used tokenizer.pad_token_id to get sequence length for gpt2 and ada[ted models (the ones pretrained on text) and used model.config.pad_token_id for CodeGPT - original models for java and python. Tha token ids according to tokenizer and model are slightly different for the adapted models causing errors. like 50256, 50257

In case of original models the model config ones are 0,1,2 for bos,eos,pad but 5025smth according to the tokenizer.



