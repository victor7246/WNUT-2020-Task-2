from __future__ import absolute_import

import sys
import os

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import pandas as pd
import numpy as np

import pickle
import joblib
from collections import Counter
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from src import data, models

import spacy

class spacy_tokenizer:
    def __init__(self,dictionary='en_core_web_sm'):
        self.nlp = spacy.load(dictionary)

    def __call__(self,text):
        return [i.text for i in self.nlp(text)]

def main(args):

    train_df = data.load_data.load_custom_text_as_pd(args.train_data,sep='\t',header=True, \
                              text_column=['Text'],target_column=['Label'])
    val_df = data.load_data.load_custom_text_as_pd(args.val_data,sep='\t', header=True, \
                          text_column=['Text'],target_column=['Label'])

    train_df = pd.DataFrame(train_df,copy=False)
    val_df = pd.DataFrame(val_df,copy=False)

    model_save_dir = args.model_save_path

    try:
        os.makedirs(model_save_dir)
    except OSError:
        pass

    train_df.labels, label2idx = data.data_utils.convert_categorical_label_to_int(train_df.labels, \
                                                         save_path=os.path.join(model_save_dir,'label2idx.pkl'))

    val_df.labels, _ = data.data_utils.convert_categorical_label_to_int(val_df.labels, \
                                                         save_path=os.path.join(model_save_dir,'label2idx.pkl'))
    idx2label = {i:w for (w,i) in label2idx.items()}

    print ("Train and val size {}, {}".format(train_df.shape[0], val_df.shape[0]))
    print (idx2label)

    nlp = spacy_tokenizer()

    print ("Vectorizer starts")
    train_tfidf, tfidf = data.preprocessing.vectorizer(train_df.words, ngram_range=(1,3), max_df=.6, max_features=None, tokenizer=nlp)
    val_tfidf = tfidf.transform(val_df.words)

    print ("Modelling")
    lr = LogisticRegression()
    lr.fit(train_tfidf, train_df.labels)

    val_pred = lr.predict(val_tfidf)

    print ("Evaluation")
    
    f1 = f1_score(val_df.labels, val_pred)
    precision = precision_score(val_df.labels, val_pred)
    recall = recall_score(val_df.labels, val_pred)

    #f1 = f1_score([idx2label[i] for i in val_df.labels], [idx2label[i] for i in val_pred])
    #precision = precision_score([idx2label[i] for i in val_df.labels], [idx2label[i] for i in val_pred])
    #recall = recall_score([idx2label[i] for i in val_df.labels], [idx2label[i] for i in val_pred])

    results_ = pd.DataFrame()
    results_['description'] = ['logistic regression with tfidf and spacy tokenization']
    results_['f1'] = [f1]
    results_['precision'] = [precision]
    results_['recall'] = [recall]

    with open(os.path.join(model_save_dir, 'vectorizer.pkl'), 'wb') as handle:
        pickle.dump(tfidf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(model_save_dir, 'model.pkl'), 'wb') as handle:
        pickle.dump(lr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if os.path.exists('../results/result.csv'):
        results = pd.read_csv('../results/result.csv')
        results = pd.concat([results, results_], axis=0)
        results.to_csv('../results/result.csv', index=False)
    else:
        results_.to_csv('../results/result.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Trainer',conflict_handler='resolve')

    parser.add_argument('--train_data', type=str, default='../data/raw/COVID19Tweet-master/train.tsv', required=False,
                        help='train data')
    parser.add_argument('--val_data', type=str, default='../data/raw/COVID19Tweet-master/valid.tsv', required=False,
                        help='validation data')

    parser.add_argument('--model_save_path', type=str, default='../models/model1/', required=False,
                        help='model save path')

    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='seed')

    args, _ = parser.parse_known_args()

    np.random.seed(args.seed)

    main(args)



