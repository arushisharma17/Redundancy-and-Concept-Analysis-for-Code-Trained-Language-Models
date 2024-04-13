# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import time
import collections
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model

cpu_cont = 16
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertModel, BertTokenizer,
                          GPT2Config, GPT2Model, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2Model, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

def get_example(item):
    url1,url2,label,tokenizer,args,cache,url_to_code=item
    if url1 in cache:
        code1=cache[url1].copy()
    else:
        try:
            code=' '.join(url_to_code[url1].split())
        except:
            code=""
        code1=tokenizer.tokenize(code)
    if url2 in cache:
        code2=cache[url2].copy()
    else:
        try:
            code=' '.join(url_to_code[url2].split())
        except:
            code=""
        code2=tokenizer.tokenize(code)
        
    return convert_examples_to_features(code1,code2,label,url1,url2,tokenizer,args,cache)


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 url1,
                 url2

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.url1=url1
        self.url2=url2
        
def convert_examples_to_features(code1_tokens,code2_tokens,label,url1,url2,tokenizer,args,cache):
    #source
    code1_tokens=code1_tokens[:args.block_size-2]
    if args.model_type == 'bert':
        code1_tokens =[tokenizer.cls_token]+code1_tokens+[tokenizer.sep_token]
        code2_tokens=code2_tokens[:args.block_size-2]
        code2_tokens =[tokenizer.cls_token]+code2_tokens+[tokenizer.sep_token]
    else:
        code1_tokens =[tokenizer.bos_token]+code1_tokens+[tokenizer.eos_token]
        code2_tokens=code2_tokens[:args.block_size-2]
        code2_tokens =[tokenizer.bos_token]+code2_tokens+[tokenizer.eos_token]  
    
    code1_ids=tokenizer.convert_tokens_to_ids(code1_tokens)
    padding_length = args.block_size - len(code1_ids)
    code1_ids+=[tokenizer.pad_token_id]*padding_length
    
    code2_ids=tokenizer.convert_tokens_to_ids(code2_tokens)
    padding_length = args.block_size - len(code2_ids)
    code2_ids+=[tokenizer.pad_token_id]*padding_length
    
    source_tokens=code1_tokens+code2_tokens
    source_ids=code1_ids+code2_ids
    return InputFeatures(source_tokens,source_ids,label,url1,url2)

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512,pool=None):
        postfix=file_path.split('/')[-1].split('.txt')[0]
        self.examples = []
        index_filename=file_path
        logger.info("Creating features from index file at %s ", index_filename)
        url_to_code={}
        with open('/'.join(index_filename.split('/')[:-1])+'/data.jsonl') as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                url_to_code[js['idx']]=js['func']

        data=[]
        data0=[]
        data1=[]
        cache={}
        f=open(index_filename)
        with open(index_filename) as f:
            for line in f:
                line=line.strip()
                url1,url2,label=line.split('\t')
                if url1 not in url_to_code or url2 not in url_to_code:
                    continue
                if label=='0':
                    label=0
                else:
                    label=1
                data.append((url1,url2,label,tokenizer, args,cache,url_to_code))
                #stratify
                if label==0:
                    data0.append((url1,url2,label,tokenizer, args,cache,url_to_code))
                elif label==1:
                    data1.append((url1,url2,label,tokenizer, args,cache,url_to_code))
                else:
                    print("corrupt sample")

        #10% train and validatioin sets but complete test set
        if 'test' not in postfix:
            #data=random.sample(data,int(len(data)*0.1))
        
            #randomly sample 10% of class 0 data
            data0=random.sample(data0,int(len(data0)*0.1))
            print(len(data0))
            #randomly sample 10% of class 1 data
            data1=random.sample(data1,int(len(data1)*0.1))
            print(len(data1))

        #balance teh dataset
        if len(data0) > len(data1):
            diff = len(data0)-len(data1)
            data0=data0[0:len(data1)]
            print(len(data0))
            print(len(data1))
        elif len(data0) < len(data1):
            data1=data1[0:len(data0)]
            print(len(data0))
            print(len(data1))

        balanced_data=data0+data1
        random.shuffle(balanced_data)
        print("balanced_data", len(balanced_data))
               
        self.examples=pool.map(get_example,tqdm(balanced_data,total=len(balanced_data)))


        if 'train' in postfix:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    #logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_tokens: {}".format([x for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))



    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label)


def load_and_cache_examples(args, tokenizer, evaluate=False,test=False,extract=False,pool=None):
    dataset = TextDataset(tokenizer, args, file_path=args.extract_data_file if extract else (args.test_data_file if test else (args.eval_data_file if evaluate else args.train_data_file)),block_size=args.block_size,pool=pool)
    return dataset

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer,pool):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_f1=0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    set_seed(args.seed)  # Added here for reproducibility (even between python 2 and 3)
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)        
            labels=batch[1].to(args.device) 
            model.train()
            loss,logits = model(inputs,labels)


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer,pool=pool,eval_when_training=True)                 
                        # Save model checkpoint

                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="",pool=None,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True,pool=pool)
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        labels=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    best_threshold=0
    best_f1=0
    for i in range(1,100):
        threshold=i/100
        y_preds=logits[:,1]>threshold
        from sklearn.metrics import recall_score
        recall=recall_score(y_trues, y_preds)
        from sklearn.metrics import precision_score
        precision=precision_score(y_trues, y_preds)   
        from sklearn.metrics import f1_score
        f1=f1_score(y_trues, y_preds) 
        if f1>best_f1:
            best_f1=f1
            best_threshold=threshold

    y_preds=logits[:,1]>best_threshold
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds)   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds)             
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
        
    }

    logger.info("***** Eval results {} *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, prefix="",pool=None,best_threshold=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = load_and_cache_examples(args, tokenizer, test=True,pool=pool)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
    #print("batches",[batch for batch in eval_dataloader])
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)        
        labels=batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss,logit = model(inputs,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    y_preds=logits[:,1]>best_threshold
    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,y_preds):
            if pred:
                f.write(example.url1+'\t'+example.url2+'\t'+'1'+'\n')
            else:
                f.write(example.url1+'\t'+example.url2+'\t'+'0'+'\n')


def extract(args, model, tokenizer, prefix="",pool=None,best_threshold=0,debug=True):
    # Loop to handle MNLI double evaluation (matched, mis-matched)

    eval_dataset = load_and_cache_examples(args, tokenizer, extract=True,pool=pool)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)
    #print("batches",[batch for batch in eval_dataloader))
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    layer_indexes = [int(x) for x in args.layers.split(",")]

    # Eval!
    logger.info("***** Running Extraction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    logger.info("  Extracting layers = %s", ",".join(map(str, layer_indexes)))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    y_trues=[]
    with open(args.output_file, "w", encoding='utf-8') as writer:
        unique_id = 0
        for batch in eval_dataloader:
            inputs = batch[0].to(args.device)
            labels=batch[1].to(args.device)
            print("inputs,labels",inputs,labels)
            with torch.no_grad():
                lm_loss,logit,hidden_states = model(inputs,labels)
                print(lm_loss, logit,hidden_states) #lm_loss=scalar, logits=list, hidden_states=tuple
                print("logit.len",len(logit)) #logit is a list of lists 4 inputs -> [[][][][]] 
                layer_outputs=hidden_states
                print("layer_outputs",layer_outputs)
                all_inputs =inputs
                print("length of layer_outputs", len(layer_outputs))
                print("len/shape of x for x in layer outputs")
                print([x.shape for x in layer_outputs])
                layer_outputs = [x.detach().cpu().numpy() for x in layer_outputs]
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())
                print("all_inputs", all_inputs, len(all_inputs))
                for example_idx in range(len(all_inputs)):
                    if debug: print("Processing example no %d" % (unique_id))
                    start_time = time.time()
                    output_json = collections.OrderedDict()

                    # Iterate over each example in the batch
                    input_ids = all_inputs[example_idx].tolist()
                    output_json["linex_index"] = unique_id
                    all_out_features = []

                    tokens = list(enumerate(tokenizer.convert_ids_to_tokens(input_ids)))  #list of tokens are being obtained from tokenizer but this is slightly different from what we have from the model now.
                    assert len(tokens) == len(input_ids)
                    if debug:
                        print("All tokens")
                        print(tokens)
                    if args.model_type == 'bert':
                        tokens = [(i,t) for i,t in tokens if t != '[PAD]']
                    elif args.model_type == 'xlnet':
                        tokens = [(i,t) for i,t in tokens if t != '<pad>']
                    elif args.model_type == 'roberta':
                        tokens = [(i,t) for i,t in tokens if t != '<pad>']
                    elif args.model_type == 'distilbert':
                        tokens = [(i,t) for i,t in tokens if t != '[PAD]']
                    elif args.model_type == 'gpt2':
                        tokens = [(i,t) for i,t in tokens if t != '<pad>']

                    #Only get CLS tokens or first token in case of clone detection, this is not relevant because it is taken care of in models.py. We only extract the first token so we don't need to put the condition here. 
                    if args.sentence_only and args.model_type == 'bert':
                        tokens = [(i,t) for i,t in tokens if (t == '[CLS]' and i==0)]
                    if args.sentence_only and args.model_type == 'roberta':
                        tokens = [(i,t) for i,t in tokens if (t == '<s>' and i==0)]
                    if args.sentence_only and args.model_type == 'auto':
                        tokens = [(i,t) for i,t in tokens if t == '<s>']
                    if args.sentence_only and args.model_type == 'xlnet':
                        tokens = [(i,t) for i,t in tokens if t == '<cls>']
                    if args.sentence_only and args.model_type == 'distilbert':
                        tokens = [(i,t) for i,t in tokens if t == '[CLS]']
                    if args.sentence_only and args.model_type == 'gpt2':
                        tokens = [(i,t) for i,t in tokens if t == '</s>'] #last token for GPT models

                    if debug:
                        print("Extracting tokens:")
                        print(tokens)
                    for token_idx, token_tuple in enumerate(tokens):  #need to enum and replace token_idx with idx so there are only 2, can remove this loop because there are activations for one token only.
                        token=token_tuple[1]
                        all_layers = []
                        for j, layer_idx in enumerate(layer_indexes): #layer_indexes=0,1,2,3,4,5,6,7,8,9,10,11,12
                            layers = collections.OrderedDict()
                            print(layer_idx, example_idx, token_idx,token)
                            layers["index"] = layer_idx
                            print(layer_outputs[layer_idx][example_idx])

                            layers["values"] = [round(x.item(),6) for x in layer_outputs[layer_idx][example_idx]] #[layer_idx][example_idx][token_idx]
                            all_layers.append(layers)
                       #print("all_layers", all_layers)
                        out_features = collections.OrderedDict()
                        out_features["token"] = token
                        out_features["layers"] = all_layers
                        all_out_features.append(out_features)
                    output_json["features"] = all_out_features
                    end_time = time.time()
                    if debug: print("Computed in %d s" %(end_time-start_time))
                    writer.write(json.dumps(output_json) + "\n")
                    end_time = time.time()
                    if debug: print("Saved in %d s" %(end_time-start_time))

                    unique_id += 1

            nb_eval_steps += 1
        logits=np.concatenate(logits,0)
        y_preds=logits[:,1]>best_threshold
    


                                                
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output file where features will be saved.")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--extract_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--do_extract", action='store_true',
                        help="Whether to run extraction on the given set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--sentence_only', action='store_true', help="For Extracting only [CLS] token embeddings")
    parser.add_argument('--layers', type=str, default='', help="Comma separated list of layers to extract. 0: Embeddings, 1-12: Normal layers")
    
    pool = multiprocessing.Pool(cpu_cont)
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//args.n_gpu
    args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=2
    config.hidden_dropout_prob=0.1

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    
    
    if args.model_type == 'gpt2':
        config.pad_token_id=tokenizer.pad_token_id  #In adapted models config pad token is None so we use teh tokenizer one but this way we have option to use either. 

    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False,pool=pool)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer,pool)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result=evaluate(args, model, tokenizer,pool=pool)
        
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer,pool=pool,best_threshold=0.5)
    
    if args.do_extract and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        extract(args, model, tokenizer,pool=pool,best_threshold=0.5)    

 
    return results


if __name__ == "__main__":
    main()

