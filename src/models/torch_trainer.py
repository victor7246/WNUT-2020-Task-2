#import .config as config
from __future__ import absolute_import

import sys
import os

from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.metric import NumpyMetric

def BCEWithLogitsLoss(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

class PLF1Score(NumpyMetric):
    def __init__(self):
        super(PLF1Score, self).__init__(f1_score)
        self.scorer = f1_score

    def forward(self, x, y):
        x = np.round(np.array(x))
        y = np.round(np.array(y))

        return self.scorer(x,y)

def test_pl_trainer(data_loader, pltrainer):
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = pltrainer(d)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            
    return fin_outputs


class PLTrainer(pl.LightningModule):

    def __init__(self, num_train_steps, model, lr, seed=42):
        super(PLTrainer, self).__init__()

        seed_everything(seed)

        self.model = model
        self.num_train_steps = num_train_steps
        self.lr = lr
        
        self.loss_fn = BCEWithLogitsLoss
        self.metric_name = 'f1'
        self.metric = PLF1Score()

        self.save_hyperparameters()

        self.print_stats()

    def print_stats(self):
        print ("[LOG] Total number of parameters to learn {}".format(sum(p.numel() for p in self.model.parameters() \
                                                                 if p.requires_grad)))

    def forward(self, x):

        return self.model(ids=x["ids"], mask=x["mask"], token_type_ids=x["token_type_ids"])
        #return self.model(ids=x["ids"], mask=x["mask"])

    def training_step(self, batch, batch_idx):
        di = batch
        ids = di["ids"]
        token_type_ids = di["token_type_ids"]
        mask = di["mask"]
        targets = di["targets"]

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        #outputs = self.model(ids=ids, mask=mask)

        loss = self.loss_fn(outputs, targets)

        metric_value = self.metric(targets, torch.sigmoid(outputs))

        tensorboard_logs = {'train_loss': loss, "train {}".format(self.metric_name): metric_value}

        return {'loss': loss, 'train_metric': metric_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        di = batch
        ids = di["ids"]
        token_type_ids = di["token_type_ids"]
        mask = di["mask"]
        targets = di["targets"]

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        #outputs = self.model(ids=ids, mask=mask)

        loss = self.loss_fn(outputs, targets)

        metric_value = self.metric(targets,torch.sigmoid(outputs))

        tensorboard_logs = {'val_loss': loss, "val {}".format(self.metric_name): metric_value}

        return {'val_loss': loss, 'val_metric': metric_value, 'log': tensorboard_logs}

    def configure_optimizers(self):

        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps)

        return [optimizer], [{'scheduler': scheduler}]

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        train_metric_mean = torch.stack([x['train_metric'] for x in outputs]).mean()
        print ("Train loss = {} Train metric = {}".format(round(train_loss_mean.detach().cpu().numpy().item(), 3),round(train_metric_mean.detach().cpu().numpy().item(), 3)))

        return {'train_loss': train_loss_mean, 'train_metric': train_metric_mean}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_metric_mean = torch.stack([x['val_metric'] for x in outputs]).mean()
        print ("val loss = {} val metric = {} ".format(round(val_loss_mean.detach().cpu().numpy().item(), 3),round(val_metric_mean.detach().cpu().numpy().item(), 3)))

        return {'val_loss': val_loss_mean, 'val_metric': val_metric_mean}

    