# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        #modify these after finetuning to get extractions
        logits=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1),output_hidden_states=True)
        hidden_states=outputs[1]
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob,hidden_states
        else:
            return prob,hidden_states
      
        
 
