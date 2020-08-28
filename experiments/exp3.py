############# Roberta, Albert, BERTweet base model runs ###########
################################################################

from __future__ import absolute_import

import sys
import os

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
import numpy as np

import pickle
from collections import Counter
from tqdm import tqdm

try:
    from dotenv import find_dotenv, load_dotenv
    import wandb
    load_dotenv(find_dotenv())
    wandb.login(key=os.environ['WANDB_API_KEY'])
    from wandb.keras import WandbCallback
    _has_wandb = True
except:
    _has_wandb = False

import tensorflow as tf
import tensorflow.keras.backend as K

import tokenizers
from transformers import TFAutoModel, AutoTokenizer, AutoConfig

from src import data, models

from sklearn.metrics import f1_score, precision_score, recall_score

def main(args):
    train_df = data.load_data.load_custom_text_as_pd(args.train_data,sep='\t',header=True, \
                              text_column=['Text'],target_column=['Label'])
    val_df = data.load_data.load_custom_text_as_pd(args.val_data,sep='\t', header=False, \
                          text_column=['Text'],target_column=['Label'])

    train_df = pd.DataFrame(train_df,copy=False)
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


    print ("Modelling")
    model = models.tf_models.transformer_base_model_cls_token(args.transformer_model_name, args.max_text_len, dropout=args.dropout)

    print (model.summary())

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)  #SGD

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='accuracy') #binary_crossentropy

    early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, \
                                         verbose=1, mode='auto', restore_best_weights=True)
    lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, \
                                          patience=3, verbose=1, mode='auto', min_lr=0.000001)
    f1callback = models.tf_utils.F1Callback(model, valX, val_outputs, filename=os.path.join(model_save_dir, 'model.h5'), patience=5)

    config = {
      'text_max_len': args.max_text_len,
      'epochs': args.epochs,
      "learning_rate": args.lr,
      "batch_size": args.train_batch_size,
      "dropout": args.dropout,
      "model_description": args.transformer_model_name.split('/')[-1]
    }

    with open(os.path.join(model_save_dir, 'config.pkl'), 'wb') as handle:
        pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)

    K.clear_session()

    if os.path.exists(os.path.join(model_save_dir, 'model.h5')) == False:
        if _has_wandb and args.wandb_logging:
            wandb.init(project='wnut-task2',config=config)
            model.fit(trainX, outputs, validation_data=(valX, val_outputs), epochs=args.epochs,\
                  batch_size=args.train_batch_size, callbacks=[early, lr, f1callback, WandbCallback()], verbose=1)
        else:
            model.fit(trainX, outputs, validation_data=(valX, val_outputs), epochs=args.epochs,\
                  batch_size=args.train_batch_size, callbacks=[early,lr, f1callback], verbose=1)

    model.load_weights(os.path.join(model_save_dir, 'model.h5'))
    #model_json = model.to_json()
    #with open(os.path.join(model_save_dir,"model.json"), "w") as json_file:
    #    json_file.write(model_json)

    val_pred = np.round(model.predict(valX))[:,0]

    print ("Evaluation")
    
    f1 = f1_score(val_df.labels, val_pred)
    precision = precision_score(val_df.labels, val_pred)
    recall = recall_score(val_df.labels, val_pred)

    #f1 = f1_score([idx2label[i] for i in val_df.labels], [idx2label[i] for i in val_pred])
    #precision = precision_score([idx2label[i] for i in val_df.labels], [idx2label[i] for i in val_pred])
    #recall = recall_score([idx2label[i] for i in val_df.labels], [idx2label[i] for i in val_pred])

    results_ = pd.DataFrame()
    results_['description'] = [args.transformer_model_name.split('/')[-1]]
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

    parser.add_argument('--model_save_path', type=str, default='../models/model3/', required=False,
                        help='model save path')

    parser.add_argument('--max_text_len', type=int, default=100, required=False,
                    help='maximum length of text')
    parser.add_argument('--dropout', type=float, default=.2, required=False,
                    help='dropout')

    parser.add_argument('--epochs', type=int, default=15, required=False,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=.00002, required=False,
                        help='learning rate')

    parser.add_argument('--train_batch_size', type=int, default=32, required=False,
                        help='train batch size')

    parser.add_argument('--wandb_logging', type=bool, default=True, required=False,
                        help='wandb logging')

    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='seed')

    #args, _ = parser.parse_known_args()
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    main(args)