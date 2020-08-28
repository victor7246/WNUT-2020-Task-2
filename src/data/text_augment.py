from tqdm import tqdm
import pandas as pd
import torch
import nlpaug.augmenter.word as naw

def augment(text,label,attack):
    '''
    e.g.
    import textattack
    import nlpaug

    textattack.augmentation.WordNetAugmenter()
    textattack.augmentation.EmbeddingAugmenter()
    nlpaug.augmenter.word.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
    nlpaug.augmenter.sentence.ContextualWordEmbsForSentenceAug(model_path='gpt2')
    nlpaug.augmenter.word.SynonymAug(aug_src='wordnet', lang='eng')

    '''

    if type(label) == list:
        print ("Augmentation does not support sequence output")
        return text, label
    else:
        augmented_texts = attack.augment(text)
        if type(augmented_texts) == list:
            labels = [label]*len(augmented_texts)

        return augmented_texts, labels

def bert_augment(text,aug_p=.2):

    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', aug_p=aug_p)
    augmented_texts = aug.augment(text)

    return augmented_texts

def translate_augment(text, lang):
    '''
    currently it supports lang = 'ru' and 'de'
    '''
    try:
        translator = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-{}.single_model'.format(lang))
        translator.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        translator.to(device)

        return translator.translate(text)

    except:
        raise ValueError("Does not support language or check torch hub path")
        return None


def translate_augment_df(df, text_column, lang):
    '''
    currently it supports lang = 'ru' and 'de'
    '''
    texts = []
    for i in tqdm(range(df.shape[0])):
        translated_txt = translate_augment(df[text_column].iloc[i])
        if translated_txt:
            texts.append(translated_txt)

    if len(texts) == df.shape[0]:
        updated_df = df.copy()
        updated_df['lang'] = lang
        updated_df[text_column] = texts

        return updated_df
    else:
        return pd.DataFrame(columns=df.columns.to_list() + ['lang'])

def masked_augment_df(df, text_column, aug_p=.2):

    texts = []
    for i in tqdm(range(df.shape[0])):
        augmented_txt = bert_augment(df[text_column].iloc[i], aug_p)
        if augmented_txt:
            texts.append(augmented_txt)

    if len(texts) == df.shape[0]:
        updated_df = df.copy()
        updated_df[text_column] = texts

        return updated_df
    else:
        return pd.DataFrame(columns=df.columns.to_list())


