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
|  BiLSTM |  0.808 |  0.854 |  0.767 |
|  RoBERTa<sub>base</sub> | 0.908  |  0.87 |  0.949 |
|  ALBERT<sub>base</sub> | 0.892  |  0.882 |  0.902 |

## Citation

