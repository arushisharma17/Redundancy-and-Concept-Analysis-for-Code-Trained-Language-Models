import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
#from transformers.modeling_bert import BertLayerNorm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_utils import PreTrainedModel

def reduce_dim_and_extract(args,config, output,bs):
        dense = nn.Linear(config.hidden_size*2, config.hidden_size,  device=args.device)
        #all hidden layers for extraction
        cls_embeddings=[]
        for layer in range(config.num_hidden_layers + 1):
            if args.model_type == "gpt2":
                cls_embedding = output[layer][:,-1,:] 
            else:
                cls_embedding = output[layer][:,0,:] #This 0 might be a problem, use all tokens then? 
            print("cls_embedding original",cls_embedding.shape,cls_embedding) # (32,768) for batch size 16
            cls_embedding = cls_embedding.reshape(-1,cls_embedding.size(-1)*2) #(batch_size, hidden_size*2)
            print("cls_embedding reshaped",cls_embedding.shape,cls_embedding) # (16,1536) for batch size 16
            cls_embedding = dense(cls_embedding) #(16,768)
            print("pooled_cls_embedding",cls_embedding.shape, cls_embedding)
            cls_embeddings.append(cls_embedding)
            # Stack the embeddings from all layers into a single tensor
        hidden_states = torch.stack(cls_embeddings, dim=0)
        print("hidden_states.shape",hidden_states.shape)
        return hidden_states


def reduce_dim_and_extract_gpt(args,config, output,bs,batch_index,last_token_id):
        dense = nn.Linear(config.hidden_size*2, config.hidden_size,  device=args.device)
        #all hidden layers for extraction
        cls_embeddings=[]
        for layer in range(config.num_hidden_layers + 1):
            if args.model_type == "gpt2":
                cls_embedding = output[layer][batch_index,last_token_id,:]
            else:
                cls_embedding = output[layer][:,0,:] #This 0 might be a problem, use all tokens then?
            print("cls_embedding original",cls_embedding.shape,cls_embedding) # (32,768) for batch size 16
            cls_embedding = cls_embedding.reshape(-1,cls_embedding.size(-1)*2) #(batch_size, hidden_size*2)
            print("cls_embedding reshaped",cls_embedding.shape,cls_embedding) # (16,1536) for batch size 16
            cls_embedding = dense(cls_embedding) #(16,768)
            print("pooled_cls_embedding",cls_embedding.shape, cls_embedding)
            cls_embeddings.append(cls_embedding)
            # Stack the embeddings from all layers into a single tensor
        hidden_states = torch.stack(cls_embeddings, dim=0)
        print("hidden_states.shape",hidden_states.shape)
        return hidden_states


class Model(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args
        

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0] #
        inputs = torch.cat((code_inputs, nl_inputs), 0)  #This is still 32
        attention_mask=inputs.ne(self.tokenizer.pad_token_id)
        print("attention_mask", attention_mask)
        print("inputs.shape",inputs.shape) #(for batch size 16, shape is (32,200))
        #outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]#OUTPUT OF POOLIN GLAYER FOR ROBERTA MODELS only cls tokens 
        if self.args.model_type == "gpt2":
            seq_length = attention_mask.sum(dim=1)-1
            print("sequence_lengths",seq_length)
            print("seq_length.shape",seq_length.shape)
            #for finetuning
            outputs = self.encoder(inputs,attention_mask=inputs.ne(self.tokenizer.pad_token_id))[0] #[0] just for last hidden state, use for finetuning
            print("shape of outputs", outputs.shape)#for batch_size 16, shape should be (32,200,768)
            output= self.encoder(inputs,attention_mask=inputs.ne(self.tokenizer.pad_token_id),output_hidden_states=True).hidden_states #use for extraction
        
            #for finetuning gpt2 models
            features=outputs   
            last_token_id = seq_length
            length = seq_length.size(dim=0)
            print("length",length)
            batch_index = torch.arange(length,device = features.device)
            print("last_token_id",last_token_id,last_token_id.shape)
            print("features.device", features.device)
            outputs=features[batch_index, last_token_id ,:] #x = features[:, -1, :]  # take </s> token (equiv. to [SEP])
            print("outputs.shape -> should be (batch_size, hidden dim (768)",outputs.shape)
            output = self.encoder(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id),output_hidden_states=True)[2] #[2] hidden states (RoBERTa and GPT)
            hidden_states=reduce_dim_and_extract_gpt(self.args, self.config, output,bs,batch_index,last_token_id)
        else:

            #for RoBERTa-like models
            outputs = self.encoder(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id))[1]#OUTPUT OF POOLIN GLAYER FOR ROBERTA MODELS only cls tokens (batch_size,hidden_dim(768))
            print("outputs.shape",outputs.shape)
                 
            if self.args.do_extract:
                #output = self.encoder(inputs, attention_mask=inputs.ne(1),output_hidden_states=True)[2] #[2] hidden states (RoBERTa and GPT)
                output = self.encoder(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id),output_hidden_states=True)[2] #[2] hidden states (RoBERTa and GPT)
                hidden_states=reduce_dim_and_extract(self.args, self.config, output,bs)

        #RoBERTa like models
        #ppooler layer outputs 
        code_vec = outputs[:bs]  #16
        nl_vec = outputs[bs:]  #16
        print("code_vec",code_vec)
        print("nl_vec",nl_vec) 
        if return_vec:
             return code_vec, nl_vec

        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 1)) #FLATTENED TO 16 FROM 32
            
        print("logits.shape", logits.shape)
        labels = labels.unsqueeze(1)
        loss = self.loss_func(logits, labels.float())
        predictions = (logits > 0.5).int()  # (Batch, )
        return loss, predictions, hidden_states
            
#if self.args.model_type=="gpt2":
         #   outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[0] #last hidden state, pooled_logits below
            #output = self.encoder(inputs, attention_mask=inputs.ne(1),output_hidden_states=True) #[2] for both RoBERTa and GPT
          #  hidden_states=output[2]

           # print("last_hidden_state output shape",outputs.shape)

           # sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)

            #LOGITS HERE ARE OUTPUTS OF LAST HIDDEN STATE
            #pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

           # code_vec = outputs[:bs]
           # nl_vec = outputs[bs:]

         #   if return_vec:
         #       return code_vec, nl_vec

          #  print("(nl_vec-code_vec, nl_vec*code_vec", nl_vec-code_vec, nl_vec*code_vec,nl_vec-code_vec.shape, nl_vec*code_vec.shape)
           # logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 1))
         #   print("logits.shape", logits.shape)

          #  output = self.encoder(inputs, attention_mask=inputs.ne(1),output_hidden_states=True)[2] #[2] for both RoBERTa and GPT

#get pooled logits using sequence lengths dep on input_ids, pad_token_ids
            #if input_ids is not None:
             #   batch_size, sequence_length = input_ids.shape[:2]

            #if self.config.pad_token_id is None:
             #   sequence_lengths = -1
            #else:
             #   if input_ids is not None:
              #      sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
               # else:
                #    sequence_lengths = -1
                 #   print("last token being used not eos, might be <pad> token")
                  #  logger.warning(
                   #     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                    #)
            #logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            #pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
