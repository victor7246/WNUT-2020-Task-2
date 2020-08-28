from __future__ import absolute_import

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from .torch_layers import MixLinear

class TransformerWithMixout(nn.Module):
    def __init__(self, pretrained_model_name, main_dropout_prob=0, mixout_prob=.7, dropout=.3, n_out=1):
        super(TransformerWithMixout, self).__init__()
        config = AutoConfig.from_pretrained(pretrained_model_name, \
                                        output_hidden_states=True, output_attentions=False)
        model = AutoModel.from_pretrained(pretrained_model_name, config=config)
        
        for i in range(model.config.num_hidden_layers):
            num = '{}'.format(i)
            for name, module in model._modules['encoder']._modules['layer']._modules[num]._modules['output']._modules.items():
                if name == 'dropout' and isinstance(module, nn.Dropout):
                    model._modules['encoder']._modules['layer']._modules[num]._modules['output']._modules[name] = nn.Dropout(main_dropout_prob)
                    #setattr(model, name, nn.Dropout(0))
                if name.split('.')[-1] == 'dense' and isinstance(module, nn.Linear):
                    target_state_dict = module.state_dict()
                    bias = True if module.bias is not None else False
                    new_module = MixLinear(module.in_features, module.out_features, 
                                           bias, target_state_dict['weight'], mixout_prob)
                    new_module.load_state_dict(target_state_dict)
                    #setattr(model, name, new_module)
                    model._modules['encoder']._modules['layer']._modules[num]._modules['output']._modules[name] = new_module

        self.base_model = model
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(model.config.hidden_size, n_out)

    def forward(self, ids, mask, token_type_ids):
        o2 = self.base_model(ids, attention_mask=mask, token_type_ids=token_type_ids)
        o2 = o2[0][:,1:,:]
        bo = self.drop(o2)
        bo = torch.mean(o2, dim=1)
        #bo = self.drop(o2)
        output = self.out(bo)

        return output
