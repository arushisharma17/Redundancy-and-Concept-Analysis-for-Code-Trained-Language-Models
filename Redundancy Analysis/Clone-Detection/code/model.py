# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class GPT2ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.config=config
        self.hidden_size=config.hidden_size
        self.batch_size=args.train_batch_size  #train for finetuning, eval for extraction        

    def forward(self, features,output,seq_length, **kwargs):
        last_token_id=seq_length[:]  #[:,1]assuming <pad> tokens are ignored due to attention mask
        print("last_token_id",last_token_id,last_token_id.shape)
        hid_size=torch.tensor(self.hidden_size).repeat(self.batch_size*2)
        print("hidden_size",hid_size,type(hid_size),hid_size.shape)

        cls_embeddings=[]
        for layer in range(self.config.num_hidden_layers + 1):
            #cls_embedding = output[layer][:, -1, :]
            cls_embedding=output[layer][torch.arange(self.batch_size*2,device=features.device), last_token_id ,:]       
            print("cls_embedding original",cls_embedding)
            cls_embedding = cls_embedding.reshape(-1,cls_embedding.size(-1)*2) #(batch_size, hidden_size*2)
            print("cls_embedding reshaped",cls_embedding.shape)
            #can just pass reshaped to dense layer, skip below steps
            cls_token1 = cls_embedding[:,:768]
            cls_token2 = cls_embedding[:, 768:]
            print("cls_token shapes",cls_token1.shape,cls_token2.shape)
            print("cls_embedding",cls_embedding.shape,cls_embedding)
            cls_embedding = self.dense(torch.cat((cls_token1, cls_token2), dim=1))
            print("pooled_cls_embedding",cls_embedding.shape, cls_embedding)
            cls_embeddings.append(cls_embedding)
        # Stack the embeddings from all layers into a single tensor
        hidden_states = torch.stack(cls_embeddings, dim=0)
        print("hidden_states.shape",hidden_states.shape)  # should be (num_layers+1, batch_size, hidden_size)
        
        #finetuning
        x=features[torch.arange(self.batch_size*2,device=features.device), last_token_id ,:] #x = features[:, -1, :]  # take </s> token (equiv. to [SEP])
        print("x before reshape", x.shape,x)
        x = x.reshape(-1,x.size(-1)*2) #This is where we need extractions from
        print("x.shape before being fed to classifier", x.shape, x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)
        self.config=config
              
    def forward(self, features,output, **kwargs):
        cls_embeddings=[]
        for layer in range(self.config.num_hidden_layers + 1):
            cls_embedding = output[layer][:, 0, :]
            print("cls_embedding original",cls_embedding)
            cls_embedding = cls_embedding.reshape(-1,cls_embedding.size(-1)*2) #(batch_size, hidden_size*2)
            print("cls_embedding reshaped",cls_embedding.shape)
            #can just pass reshaped to dense layer, skip below steps
            cls_token1 = cls_embedding[:,:768]
            cls_token2 = cls_embedding[:, 768:]
            print("cls_token shapes",cls_token1.shape,cls_token2.shape)
            print("cls_embedding",cls_embedding.shape,cls_embedding)
            cls_embedding = self.dense(torch.cat((cls_token1, cls_token2), dim=1))
            print("pooled_cls_embedding",cls_embedding.shape, cls_embedding)
            cls_embeddings.append(cls_embedding)
            # Stack the embeddings from all layers into a single tensor
        hidden_states = torch.stack(cls_embeddings, dim=0)
        print("hidden_states.shape",hidden_states.shape)  # should be (num_layers+1, batch_size, hidden_size)

        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        print("x before reshape", x.shape,x)  #(batch_size*2,hidden_size)
        x = x.reshape(-1,x.size(-1)*2) #reshaped so each code snippet is treated as a single input
        print("x.shape before being fed to classifier", x.shape, x) #(batch_size,hidden_size)

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(activations)
        x = self.dropout(x)
        x = self.out_proj(x)
         
        return x,hidden_states
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.classifier1=RobertaClassificationHead(config)
        self.classifier2=GPT2ClassificationHead(config,args)
        
    def forward(self, input_ids=None,labels=None): 
        print("shape of original input_ids",input_ids.shape, input_ids[:3])
        input_ids=input_ids.view(-1,self.args.block_size) #here it splits inputs into two
        print("input_ids",input_ids) #inputs in batch=4, input_ids=8, outputs=8, labels =4?
        print("input_ids.shape",input_ids.shape)
        print("pad_token_ids",self.config.pad_token_id, self.tokenizer.pad_token_id)   
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        print("attention_mask",attention_mask)
        
        #for finetuning
        outputs = self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(self.tokenizer.pad_token_id))[0] #[0] just for last hidden state, use for finetuning
        print("shape of outputs", outputs.shape)
        output= self.encoder(input_ids= input_ids,attention_mask=input_ids.ne(self.tokenizer.pad_token_id),output_hidden_states=True).hidden_states #use for extraction

        if self.args.model_type == "gpt2": 
            seq_length = attention_mask.sum(dim=1)-1 
            print("sequence_lengths",seq_length)
            print("seq_length.shape",seq_length.shape)
            logits=self.classifier2(outputs,output,seq_length) #GPT2ClassificationHead
        else: 
            logits,hidden_states=self.classifier1(outputs,output)  #RobertaClassificationHEAD


        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if self.args.do_extract==True:
                return loss,prob,hidden_states
            else:
                return loss,prob
        else:
            if self.args.do_extract==True:
                return prob,hidden_states
            else:
                return prob
      
        

        


