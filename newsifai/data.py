import re
from config import config
from nltk.stem import PorterStemmer
import pandas as pd
from sklearn.model_selection import train_test_split

stemmer = PorterStemmer()

def clean_text(text, lower = True, stem = False, stopwords = config.STOPWORDS):
    ''' Function cleans a news texts.

    Args:
        text (str) : news text to clean
        df (pd.DataFrame): data to preprocess
        lower (bool) : if true, convert strings to lower case. Defaults to True
        stem (bool) : if true, stemmetize texts. Defaults to False
        stopwords (List) : List of stopwords to remove. Defaults to stopwords defined in config

    Returns:
        str : clean text 
    '''
    if lower:
        text = text.lower()
    #remove stopwords
    if len(stopwords):
            pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
            text = pattern.sub('', text)
    # Spacing and filters
    text = re.sub(
        r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text
    )  # add spacing between objects to be filtered
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends

    # Stemming
    if stem:
        text = " ".join([stemmer.stem(word, to_lowercase=lower) for word in text.split(" ")])

    return text


def preprocess(df : pd.DataFrame, lower : bool, stem :bool) -> pd.DataFrame:
    '''Process Dataframe.
    
    Args:
        df (pd.DataFrame): data to preprocess
        lower (bool) : if true, convert strings to lower case
        stem (bool) : if true, stemmetize texts

    Returns:
        pd.DataFrame : preprocessed dataframe

    
    '''
    df['news'] = df['Title'].map(str) + '. ' + df['Excerpt'].map(str)
    df.news = df.news.apply(clean_text, lower=lower, stem=stem)
    return df

def get_data_splits(X : pd.Series, y : pd.Series, test_size : float= 0.2):
    """Generate balanced data splits.
    Args:
        X (pd.Series) : Features data
        y (pd.Series): Targets
        test_size (float) : percentage of data for validation. Defaults to 0.2
        

    Returns:
        np.ndarray : data splits
    '''"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size)
    return X_train,X_val, y_train, y_val
