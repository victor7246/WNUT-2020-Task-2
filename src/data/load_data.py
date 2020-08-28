import pandas as pd

def load_custom_text_as_pd(filepath, sep='\t', header=True, text_column=[], target_column=[], encoding='utf8'):
    lines = open(filepath,'r', encoding=encoding).readlines()
    lines = [line.replace('\n','') for line in lines]
    
    if header:
        headers = lines[0].split(sep)
    else:
        headers = ["col_{}".format(i) for i in range(1,len(lines[0].split(sep))+1)]

    df = pd.DataFrame()

    for i, col in enumerate(headers):
        if header:
            df[col] = [line.split(sep)[i] for line in lines[1:]]
        else:
            df[col] = [line.split(sep)[i] for line in lines]

    if len(text_column) == 1:
        df = df.rename({text_column[0]:'words'}, axis=1)

    if len(target_column) == 1:
        df = df.rename({target_column[0]:'labels'}, axis=1)
        
    return df