# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch    
class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.hash_layer = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Linear(1024, 768)
        )
        self.tanh = nn.Tanh()
      
    def forward(self, code_inputs=None, attn_mask=None,position_idx=None, nl_inputs=None, hash_scale=None, retrieval_model=False):
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask = nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
            nodes_to_token_mask = nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask,inputs_embeddings)
            inputs_embeddings = inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]
            code_dense_repr = self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[1]
            code_hash_repr = self.hash_layer(code_dense_repr)
            code_hash_repr = self.tanh(code_hash_repr * hash_scale)
            if retrieval_model == True:
                code_hash_repr = torch.sign(code_hash_repr)
            return code_hash_repr
        else:
            nl_dense_repr = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
            nl_hash_repr = self.hash_layer(nl_dense_repr)
            nl_hash_repr = self.tanh(nl_hash_repr * hash_scale)
            if retrieval_model == True:
                nl_hash_repr = torch.sign(nl_hash_repr)
            return nl_hash_repr


class Second_Stage_Model(nn.Module):
    def __init__(self, encoder):
        super(Second_Stage_Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, attn_mask=None, position_idx=None, nl_inputs=None):
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
            return self.encoder(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[1]
        else:
            return self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[1]
