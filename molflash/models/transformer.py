import torch
import torch.nn as nn
import torch.optim as optim

import math
import random
import numpy as np
from time import time


class PointWiseFeedForwardLayer(nn.Module):
    
    def __init__(self,hidden_dimension,pointwise_ff_dim,dropout):
        
        super().__init__()
        
        self.fully_connected_1 = nn.Linear(hidden_dimension,pointwise_ff_dim)
        
        self.fully_connected_2 = nn.Linear(pointwise_ff_dim,hidden_dimension)
        
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self,inp):        
        fc1_op = self.fully_connected_1(inp)
        fc1_op = torch.relu(fc1_op)        
        fc2_op = self.fully_connected_2(fc1_op)        
        return  fc2_op


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self,hidden_dimension,num_attention_heads,
                dropout):
        super().__init__()
        
        assert hidden_dimension % num_attention_heads == 0
        
        self.hidden_dimension = hidden_dimension
        self.num_attention_heads = num_attention_heads
        self.head_dimension = hidden_dimension // num_attention_heads
        
        self.W_q = nn.Linear(hidden_dimension,hidden_dimension)
        self.W_k = nn.Linear(hidden_dimension,hidden_dimension)
        self.W_v = nn.Linear(hidden_dimension,hidden_dimension)
        
        self.W_o = nn.Linear(hidden_dimension,hidden_dimension)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dimension]))
       
    
    def split_heads(self,item,batch_size):
        item = item.view(batch_size,-1,self.num_attention_heads,self.head_dimension)
        item = item.permute(0,2,1,3)
        return item
    
    def forward(self,query,key,value,mask = None):
        
        batch_size = query.shape[0]
        
        Q = self.W_q(query) #Q,query shape is (bsiz,qlen,hdim)
        K = self.W_k(key)  #K,key shape is (bsiz,klen,hdim)
        V = self.W_v(value)  #V,value shape is (bsiz,vlen,hdim)
        
        Q = self.split_heads(Q,batch_size) #Q shape(bsiz,n_attn_heads,qlen,head_dim)
        K = self.split_heads(K,batch_size) #K shape(bsiz,n_attn_heads,klen,head_dim)
        V = self.split_heads(V,batch_size) #V shape(bsiz,n_attn_heads,vlen,head_dim)

        energy = torch.matmul(Q,K.permute(0,1,3,2)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e15)
        
        attention = torch.softmax(energy,dim = -1)           
        attention_scored_value = torch.matmul(self.dropout(attention),V)
        attention_scored_value = attention_scored_value.permute(0,2,1,3).contiguous()
        attention_scored_value = attention_scored_value.view(batch_size,-1,self.hidden_dimension)
        attention_contexts_Z = self.W_o(attention_scored_value)
        return attention_contexts_Z,attention


class EncoderBlock(nn.Module):
    def __init__(self,hidden_dimension,num_attention_heads,
                      pointwise_ff_dim,dropout):
        
        super().__init__()
        
        self.attention_layer_norm = nn.LayerNorm(hidden_dimension)
        self.feedForward_layer_norm = nn.LayerNorm(hidden_dimension)
        self.selfAttention = MultiHeadSelfAttention(hidden_dimension,num_attention_heads,dropout)
        self.pointWise_feedForward = PointWiseFeedForwardLayer(hidden_dimension,pointwise_ff_dim,dropout)                                    
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self,source,source_mask):
        
        #source(x) = (bsiz,slen,hdim)
        #sourcemask = (bsize,1,1,slen)
        
        attention_contexts,attention_scores = self.selfAttention(source,source,source,source_mask)
        attention_contexts = self.dropout(attention_contexts)
        source = self.attention_layer_norm(source + attention_contexts)
        attention_contexts = self.pointWise_feedForward(source)
        attention_contexts = self.dropout(attention_contexts)
        source = self.feedForward_layer_norm(source + attention_contexts)
        return source



class Encoder(nn.Module):
    def __init__(self,input_dimension,hidden_dimension,
                 number_encoder_layers,num_attention_heads,pointwise_ff_dim,
                 dropout,max_length = 100):
        super().__init__()
        
        self.token_embeddings = nn.Embedding(input_dimension,hidden_dimension)
        self.positional_embeddings = nn.Embedding(max_length,hidden_dimension)
        
        
        self.encoder_blocks = nn.ModuleList([EncoderBlock(hidden_dimension,num_attention_heads,
                                                  pointwise_ff_dim,dropout)
                                     for _ in range(number_encoder_layers)])
        
        self.dropout = nn.Dropout(dropout)
    
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dimension]))     
        
        
    def forward(self,source,source_mask):
        
        #src = (bsiz,slen)
        #src_mask = (bsiz,1,1,slen)
        
        batch_size = source.shape[0]
        source_length = source.shape[1]
        
        positions = torch.arange(0,source_length).unsqueeze(0)
        positions = positions.repeat(batch_size,1)                       
        source = self.token_embeddings(source)*self.scale
        source = source + self.positional_embeddings(positions)
        source = self.dropout(source)
        
        for encoder_block in self.encoder_blocks:
            source = encoder_block(source,source_mask)
        
        return source




class DecoderBlock(nn.Module):
    
    def __init__(self,hidden_dimension,num_attention_heads,
                pointwise_ff_dim,dropout):
        
        super().__init__()
        
        self.decoder_self_attn_layer_norm = nn.LayerNorm(hidden_dimension)
        self.encoder_cross_attn_layer_norm = nn.LayerNorm(hidden_dimension)
        self.feedForward_layer_norm = nn.LayerNorm(hidden_dimension)
        
        self.decoder_self_attention = MultiHeadSelfAttention(hidden_dimension,num_attention_heads,dropout)
                                                                    
        self.encoder_cross_attention = MultiHeadSelfAttention(hidden_dimension,num_attention_heads,dropout)
        
        self.pointWise_feedForward = PointWiseFeedForwardLayer(hidden_dimension,pointwise_ff_dim,dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self,target,encoder_source,source_mask,target_mask):

        #target = (bsiz,tlen,hid_dim)
        #encoder_src = (bsiz,slen,hdim)
        #target_mask = (bsiz,1,tlen,tlen)
        #src_mask = (bsiz,1,1,slen)
        
        tar_self_attention_contexts,tar_self_attention_scores = self.decoder_self_attention(target,target,target,target_mask)

        tar_self_attention_contexts = self.dropout(tar_self_attention_contexts)
        target = self.decoder_self_attn_layer_norm(target + tar_self_attention_contexts)
        tar_cross_attention_contexts,tar_cross_attention_scores = self.encoder_cross_attention(target,encoder_source,encoder_source,source_mask)
        tar_cross_attention_contexts = self.dropout(tar_cross_attention_contexts)
        
        target = self.encoder_cross_attn_layer_norm(target + tar_cross_attention_contexts)
        tar_cross_attention_contexts = self.pointWise_feedForward(target)
        
        tar_cross_attention_contexts = self.dropout(tar_cross_attention_contexts)
        target = self.feedForward_layer_norm(target + tar_cross_attention_contexts)
        
        return target,tar_cross_attention_scores



class Decoder(nn.Module):
    
    
    def __init__(self,output_dimension,hidden_dimension,
                number_decoder_layers,num_attention_heads,
                pointwise_ff_dim,dropout,
                max_length = 100):
        
        super().__init__()
                
        self.token_embeddings = nn.Embedding(output_dimension,hidden_dimension)
        self.positional_embeddings = nn.Embedding(max_length,hidden_dimension)
        
        
        self.decoder_blocks = nn.ModuleList([DecoderBlock(hidden_dimension,
                                                         num_attention_heads,
                                                         pointwise_ff_dim,
                                                         dropout)
                                           for _ in range(number_decoder_layers)])
        
        
        self.fully_connected_op = nn.Linear(hidden_dimension,output_dimension)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dimension]))
    
    
    
    def forward(self,target,encoder_src,source_mask,target_mask):
        
        batch_size = target.shape[0]
        target_len = target.shape[1]
 
        positions = torch.arange(0,target_len).unsqueeze(0)
        positions = positions.repeat(batch_size,1)   
        
        target = self.token_embeddings(target)*self.scale
        
        target = target + self.positional_embeddings(positions)        
        
        target = self.dropout(target)        
        for decoder_block in self.decoder_blocks:
            target,attention = decoder_block(target,encoder_src,
                                             source_mask,target_mask)
                    
        output = self.fully_connected_op(target)        
        return output,attention


class Seq2SeqModel(nn.Module):
    
    def __init__(self,encoder_stack,decoder_stack,
                source_pad_idx,target_pad_idx):
        
        super().__init__()
        
        self.encoder_stack = encoder_stack
        self.decoder_stack = decoder_stack
        self.source_pad_index = source_pad_idx
        self.target_pad_index = target_pad_idx
        
    def get_source_mask(self,source):
        
        source_mask = (source != self.source_pad_index)
        source_mask = source_mask.unsqueeze(1)
        source_mask = source_mask.unsqueeze(2)

        return source_mask

    def get_target_mask(self,target):
        target_pad_mask = (target != self.target_pad_index).unsqueeze(1).unsqueeze(2)        
        target_length = target.shape[1]
        target_subsequent_mask = torch.tril(torch.ones((target_length,target_length))).bool()
        final_target_mask = target_subsequent_mask & target_pad_mask        
        return final_target_mask
    
    
    def forward(self,source,target):
        
        #source = (bsize,slen)
        #target = (bsize,tlen)
        
        source_mask = self.get_source_mask(source)
        target_mask = self.get_target_mask(target)
        
        #source_mask = (bsize,1,1,slen)
        #target_mask = (bsize,1,tlen,tlen)
        
        encoder_output = self.encoder_stack(source,source_mask)
        #encoder_op = (bsize,slen,hdim)
        
        decoder_output,decoder_attention = self.decoder_stack(target,encoder_output,source_mask,target_mask)
        
        #decoder_output = (bsiz,tlen,output_dim)
        #decoder_attention = (bsize,num_attention_heads,tlen,slen)
        
        # return decoder_output,decoder_attention
        return decoder_output