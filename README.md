# WNUT-2020-Task-2
Code for team datamafia's submission at WNUT 2020 Task 2: Identification of informative COVID-19 English Tweets.

## Dataset
The dataset contains tweets labelled with INFORMATIVE or, NONINFORMATIVE in the context of Covid-19. Datasets can be found at https://github.com/VinAIResearch/COVID19Tweet

## Requirements
Python 3.6 is needed to run the code. Install the dependencies using
	
	$ pip install -r requirements.txt

## How to run the experiments
All the experiments are stored inside ./experiments/ folder. Use --help to get the command line arguments expected by the code.

	$ python exp3.py --help

execute RoBERTa baseline by running

	$python exp3.py \
	 --train_data ../data/raw/COVID19Tweet/train.tsv \
	 --val_data ../data/raw/COVID19Tweet/valid.tsv \
	 --transformer_model_name roberta-base \
	 --model_save_path ../models/model_roberta/ \
	 --max_text_len 100 \
	 --dropout .2 \
	 --epochs 10 \
	 --lr .00002 \
	 --train_batch_size 32 \
	 --wandb_logging True \
	 --seed 42

## Results

|  Model | F1  | Precision  | Recall  |
|---|---|---|---|
| Logistic Regression   | 0.762  | 0.806  | 0.722  |
| BiLSTM |  0.808 |  0.854 |  0.767 |
| BERT-base-cased | 0.882  |  0.893 |  0.870 |
| RoBERTa<sub>base</sub> | <b>0.908</b>  |  0.872 |  0.949 |
| XLNet-base-cased | 0.894  |  0.839 |  <b>0.957</b> |
| ALBERT<sub>base</sub> | 0.892  |  0.882 |  0.902 |
| BERTweet | 0.899  |  <b>0.888</b> |  0.911 |

## Citation
```tex
@inproceedings{sengupta-2020-datamafia,
    title = DATAMAFIA at WNUT-2020 Task 2: A Study of Pre-trained Language Models along with Regularization Techniques for Downstream Tasks,
    author = Sengupta, Ayan,
    booktitle = Proceedings of the Sixth Workshop on Noisy User-generated Text (W-NUT 2020),
    month = nov,
    year = 2020,
    address = Online,
    publisher = Association for Computational Linguistics,
    url = https://www.aclweb.org/anthology/2020.wnut-1.51,
    pages = 371--377
}
```
