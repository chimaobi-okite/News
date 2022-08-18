from typing import Dict
from argparse import Namespace
from newsifai import utils, data, evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd
from pathlib import Path
from config import config
from config.config import logger
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression


def train(args : Namespace, df : pd.DataFrame) -> Dict:
    """Train model on data.
    Args:
        args (Namespace): arguments to use for training.
        df (pd.DataFrame): data for training.
        
    
    Returns:
        Dict: artifacts from the run.
    """
    cat2index = utils.load_dict(Path(config.DATA_DIR, 'catmapping.json'))
    utils.set_seed()
    df = data.preprocess(df, lower = True, stem = args.stem)
    X_train,X_val, y_train, y_val = data.get_data_splits(df.news, df.Category.map(cat2index))

    vectorizer = TfidfVectorizer(analyzer= args.analyzer, ngram_range= (2,args.ngram_max_range))
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    over_sampler = RandomOverSampler(sampling_strategy= 'all')
    X_over, y_over = over_sampler.fit_resample(X_train, y_train)

    model = LogisticRegression(C = args.C, random_state = args.random_state, max_iter= 1000)
    model.fit(X_over, y_over)

    # inspect performance
    y_pred = model.predict(X_val)
    performance = evaluate.evaluate(y_true= y_val, y_pred= y_pred)
    args = vars(args)
    logger.info(json.dumps(performance, indent=2))

    return {
            "args": args,
            "vectorizer": vectorizer,
            "model": model,
            "performance": performance
        }