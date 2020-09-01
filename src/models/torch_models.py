from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .torch_layers import MixLinear

class TransformerWithMixout(nn.Module):
    def __init__(self, pretrained_model_name, mixout_prob=.7, dropout=0, n_out=1):
        super(TransformerWithMixout, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name, \
                                        output_hidden_states=True, output_attentions=False)
        config.hidden_dropout_prob = dropout
        self.base_model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        
        for i in range(self.base_model .config.num_hidden_layers):
            num = '{}'.format(i)
            for name, module in self.base_model._modules['encoder']._modules['layer']._modules[num]._modules['intermediate']._modules.items():
                if name == 'dropout' and isinstance(module, nn.Dropout):
                    self.base_model._modules['encoder']._modules['layer']._modules[num]._modules['intermediate']._modules[name] = nn.Dropout(dropout)
                    #setattr(self.base_model , name, nn.Dropout(0))
                if name.split('.')[-1] == 'dense' and isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(module.in_features, module.out_features, 
                                           bias, target_state_dict['weight'], mixout_prob)
                    new_module.load_state_dict(target_state_dict)
                    #setattr(self.base_model , name, new_module)
                    self.base_model._modules['encoder']._modules['layer']._modules[num]._modules['intermediate']._modules[name] = new_module

            for name, module in self.base_model._modules['encoder']._modules['layer']._modules[num]._modules['output']._modules.items():
                if name == 'dropout' and isinstance(module, nn.Dropout):
                    self.base_model._modules['encoder']._modules['layer']._modules[num]._modules['output']._modules[name] = nn.Dropout(dropout)
                    #setattr(self.base_model , name, nn.Dropout(0))
                if name.split('.')[-1] == 'dense' and isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(module.in_features, module.out_features, 
                                           bias, target_state_dict['weight'], mixout_prob)
                    new_module.load_state_dict(target_state_dict)
                    #setattr(self.base_model, name, new_module)
                    self.base_model._modules['encoder']._modules['layer']._modules[num]._modules['output']._modules[name] = new_module

        #self.base_model = model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.base_model.config.hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o2 = o2[0][:,0,:]
        bo = self.drop(o2)
        #bo = torch.mean(o2, dim=1)
        #bo = self.drop(o2)
        output = self.out(bo)

        return output

class Transformer(nn.Module):
    def __init__(self, pretrained_model_name, dropout=.1, n_out=1):
        super(Transformer, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name, \
                                        output_hidden_states=True, output_attentions=False)
        config.hidden_dropout_prob = dropout
        self.base_model = AutoModel.from_pretrained(pretrained_model_name, config=config)

        #self.base_model = model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.base_model.config.hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o2 = o2[0][:,0,:]
        bo = self.drop(o2)

        output = self.out(bo)

        return output

class TransformerMultiSample(nn.Module):
    def __init__(self, pretrained_model_name, device, dropout=.2, multi_sample_dropout_count=5, n_out=1):
        super(TransformerMultiSample, self).__init__()
        self.device = device
        
        config = AutoConfig.from_pretrained(pretrained_model_name, \
                                        output_hidden_states=True, output_attentions=False)
        config.hidden_dropout_prob = dropout
        self.base_model = AutoModel.from_pretrained(pretrained_model_name, config=config).to(device)

        #self.base_model = model
        self.multi_sample_dropout_count = multi_sample_dropout_count
        self.drops = [nn.Dropout(np.random.random()*dropout).to(device) for i in range(multi_sample_dropout_count)]
        self.outs = [nn.Linear(self.base_model.config.hidden_size, n_out).to(device) for i in range(multi_sample_dropout_count)]
        self.final_out = nn.Linear(multi_sample_dropout_count, 1, bias=False).to(device)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids.to(self.device), attention_mask=mask.to(self.device), token_type_ids=token_type_ids.to(self.device))
        o2 = o2[0][:,0,:]
        bo = torch.cat([self.outs[i](self.drops[i](o2)) for i in range(self.multi_sample_dropout_count)], -1)
        
        output = self.final_out(bo)

        return output
