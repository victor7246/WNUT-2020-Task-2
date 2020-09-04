############# Torch Mixout ###########
################################################################

from __future__ import absolute_import

import sys
import os

import argparse
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import re
import pandas as pd
import numpy as np

import pickle
from collections import Counter
from tqdm import tqdm
from glob import glob

try:
    from dotenv import find_dotenv, load_dotenv
    import wandb
    load_dotenv(find_dotenv())
    wandb.login(key=os.environ['WANDB_API_KEY'])
    from wandb.keras import WandbCallback
    _has_wandb = True
except:
    _has_wandb = False

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.metrics.metric import NumpyMetric
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

import tokenizers
from transformers import TFAutoModel, AutoTokenizer, AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup

from src import data, models

from sklearn.metrics import f1_score, precision_score, recall_score

def main(args):
    train_df = data.load_data.load_custom_text_as_pd(args.train_data,sep='\t',header=True, \
                              text_column=['Text'],target_column=['Label'])
    val_df = data.load_data.load_custom_text_as_pd(args.val_data,sep='\t', header=False, \
                          text_column=['Text'],target_column=['Label'])

    train_df = pd.DataFrame(train_df,copy=False)
    if 'aug_p' in train_df.columns:
        train_df.aug_p = train_df.aug_p.astype(float)
        train_df = train_df[train_df.aug_p <= args.aug_p_rate].reset_index(drop=True)
        train_df = train_df.drop(['aug_p'], axis=1)

    val_df = pd.DataFrame(val_df,copy=False)
    val_df.columns = train_df.columns

    model_save_dir = args.model_save_path

    try:
        os.makedirs(model_save_dir)
    except OSError:
        pass

    train_df.labels, label2idx = data.data_utils.convert_categorical_label_to_int(train_df.labels, \
                                                         save_path=os.path.join(model_save_dir,'label2idx.pkl'))

    val_df.labels, _ = data.data_utils.convert_categorical_label_to_int(val_df.labels, \
                                                         save_path=os.path.join(model_save_dir,'label2idx.pkl'))

    print ("Tokenization")

    if 'bertweet' in args.transformer_model_name.lower():
        bertweettokenizer = True
    else:
        bertweettokenizer = False

    if bertweettokenizer == True:
        tokenizer = data.custom_tokenizers.BERTweetTokenizer(args.transformer_model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.transformer_model_name)

    trainX = data.data_utils.compute_transformer_input_arrays(train_df, 'words', tokenizer, args.max_text_len, bertweettokenizer)
    valX = data.data_utils.compute_transformer_input_arrays(val_df, 'words', tokenizer, args.max_text_len, bertweettokenizer)

    outputs = data.data_utils.compute_output_arrays(train_df, 'labels')
    val_outputs = data.data_utils.compute_output_arrays(val_df, 'labels')

    outputs = outputs[:,np.newaxis]
    val_outputs = val_outputs[:,np.newaxis]

    train_dataset = data.data_utils.TorchDataLoader(trainX[0], trainX[1], trainX[2], outputs)
    val_dataset = data.data_utils.TorchDataLoader(valX[0], valX[1], valX[2], val_outputs)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size)

    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.train_batch_size)

    print ("Train and val loader length {} and {}".format(len(train_data_loader), len(val_data_loader)))

    print ("Modelling")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ("Device: {}".format(device))

    model = models.torch_models.TransformerAvgPool(args.transformer_model_name, dropout=args.dropout)

    print (model)

    config = {
      'text_max_len': args.max_text_len,
      'epochs': args.epochs,
      "learning_rate": args.lr,
      "batch_size": args.train_batch_size,
      "dropout": args.dropout,
      "model_description": args.transformer_model_name + ' mean of all tokens from last 4 layers'
    }

    with open(os.path.join(model_save_dir, 'config.pkl'), 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
    checkpoint_callback = ModelCheckpoint(
                filepath=args.model_save_path,
                save_top_k=1,
                verbose=True,
                monitor='val_metric',
                mode='max',
                save_weights_only=True
                )

    earlystop = EarlyStopping(
                monitor='val_metric',
                patience=5,
               verbose=False,
               mode='max'
               )

    if args.lightning == 'true':
        if _has_wandb and args.wandb_logging == 'true':
            wandb.init(project="wnut-task2-regularization",config=config)
            wandb_logger = WandbLogger()

            if torch.cuda.is_available():
                trainer = Trainer(gpus=1, max_epochs=args.epochs, logger=wandb_logger, \
                                    checkpoint_callback=checkpoint_callback, callbacks=[earlystop])
            else:
                trainer = Trainer(max_epochs=args.epochs, logger=wandb_logger, \
                                    checkpoint_callback=checkpoint_callback, callbacks=[earlystop])

        else:
            if torch.cuda.is_available():
                trainer = Trainer(gpus=1, max_epochs=args.epochs, \
                                    checkpoint_callback=checkpoint_callback, callbacks=[earlystop])
            else:
                trainer = Trainer(max_epochs=args.epochs, \
                                    checkpoint_callback=checkpoint_callback, callbacks=[earlystop])        

        num_train_steps = int(len(train_data_loader) * args.epochs)

        pltrainer = models.torch_trainer.PLTrainer(num_train_steps, model, args.lr, args.l2, seed=args.seed)

        #pltrainer = Trainer(resume_from_checkpoint=glob(args.model_save_path+'*.ckpt')[0])

        checkpoints = glob(args.model_save_path+'*.ckpt')

        if len(checkpoints) > 0:
            best_checkpoint = torch.load(checkpoints[0])
            updated_checkpoint_state = OrderedDict([('.'.join(key.split('.')[1:]), v) for key, v in best_checkpoint['state_dict'].items()])
            #updated_checkpoint_state = OrderedDict([('.'.join(key.split('.')[1:]), v) for key, v in best_checkpoint.items()])
            model.load_state_dict(updated_checkpoint_state)
            val_pred = models.torch_trainer.test_pl_trainer(val_data_loader, model)

            val_pred = np.round(np.array(val_pred))[:,0]

        else:
            trainer.fit(pltrainer, train_data_loader, val_data_loader)
            checkpoints = glob(args.model_save_path+'*.ckpt')
            best_checkpoint = int(checkpoints[0].split('/')[-1].split('.')[0].split('=')[-1])
            val_pred = pltrainer.val_predictions[best_checkpoint].detach().cpu().numpy()

            val_pred = np.round(np.array(val_pred))
            #best_checkpoint = torch.load(checkpoints[0])
            #updated_checkpoint_state = OrderedDict([('.'.join(key.split('.')[1:]), v) for key, v in best_checkpoint['state_dict'].items()])
            #updated_checkpoint_state = OrderedDict([('.'.join(key.split('.')[1:]), v) for key, v in best_checkpoint.items()])
            #model.load_state_dict(updated_checkpoint_state)

    else:

        trainer = models.torch_trainer.BasicTrainer(model, train_data_loader, val_data_loader, device)

        param_optimizer = list(trainer.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        num_train_steps = int(len(train_data_loader) * args.epochs)

        optimizer = AdamW(optimizer_parameters, lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

        use_wandb = _has_wandb and (args.wandb_logging == 'true')

        if os.path.exists(os.path.join(args.model_save_path, 'model.bin')) == False:
            trainer.train(args.epochs, optimizer, scheduler, args.model_save_path, config, args.l2,  \
                early_stopping_rounds=5, use_wandb=use_wandb, seed=args.seed)
        else:
            model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'model.bin')))

        model.load_state_dict(torch.load(os.path.join(args.model_save_path, 'model.bin')))

        val_pred = models.torch_trainer.test_torch(val_data_loader, model, device)

        val_pred = np.round(np.array(val_pred))[:,0]

    print ("Evaluation")
    
    f1 = f1_score(val_df.labels, val_pred)
    precision = precision_score(val_df.labels, val_pred)
    recall = recall_score(val_df.labels, val_pred)

    #f1 = f1_score([idx2label[i] for i in val_df.labels], [idx2label[i] for i in val_pred])
    #precision = precision_score([idx2label[i] for i in val_df.labels], [idx2label[i] for i in val_pred])
    #recall = recall_score([idx2label[i] for i in val_df.labels], [idx2label[i] for i in val_pred])

    results_ = pd.DataFrame()
    results_['description'] = [config['model_description']]
    results_['f1'] = [f1]
    results_['precision'] = [precision]
    results_['recall'] = [recall]

    print (results_.iloc[0])
    
    if os.path.exists('../results/result.csv'):
        results = pd.read_csv('../results/result.csv')
        results = pd.concat([results, results_], axis=0)
        results.to_csv('../results/result.csv', index=False)
    else:
        results_.to_csv('../results/result.csv', index=False)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')

    parser.add_argument('--train_data', type=str, default='../data/raw/COVID19Tweet/train.tsv', required=False,
                        help='train data')
    parser.add_argument('--val_data', type=str, default='../data/raw/COVID19Tweet/valid.tsv', required=False,
                        help='validation data')

    parser.add_argument('--transformer_model_name', type=str, default='roberta-base', required=False,
                        help='transformer model name')

    parser.add_argument('--model_save_path', type=str, default='../models/model_final/', required=False,
                        help='model save path')

    parser.add_argument('--max_text_len', type=int, default=100, required=False,
                    help='maximum length of text')

    parser.add_argument('--dropout', type=float, default=0.2, required=False,
                    help='dropout')

    parser.add_argument('--epochs', type=int, default=15, required=False,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=.00002, required=False,
                        help='learning rate')

    parser.add_argument('--train_batch_size', type=int, default=32, required=False,
                        help='train batch size')

    parser.add_argument('--lightning', type=str, default='false', required=False,
                        help='Pytorch Lightning')

    parser.add_argument('--wandb_logging', type=str, default='false', required=False,
                        help='wandb logging')

    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='seed')

    #args, _ = parser.parse_known_args()
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)