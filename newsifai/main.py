from typing import List
from config.config import logger
import pandas as pd
import os
import json
import joblib
from config import config
from pathlib import Path
from urllib.request import urlopen
from argparse import Namespace
from newsifai import utils, train, predict
import typer
app = typer.Typer()

@app.command()
def etl_data():
    ''' Extract data from online storage, transform and save to local storage'''
    train = pd.read_csv(config.TRAIN_URL)
    test = pd.read_csv(config.TEST_URL)
    cat_mapping = json.loads(urlopen(config.MAPPING_URL).read())

    train = train.dropna(axis = 0)
    test = test.dropna(axis = 0)

    utils.save_dict(train.to_dict(orient='records'), Path(config.DATA_DIR, 'train.json'))
    utils.save_dict(test.to_dict(orient='records'), Path(config.DATA_DIR, 'test.json'))
    utils.save_dict(cat_mapping, Path(config.DATA_DIR, 'catmapping.json'))

    logger.info("✅ ETL on data is complete!")

@app.command()
def train_model():
    '''
    Trains a model and saves its artifacts
    '''
    args_fp = Path(config.CONFIG_DIR, 'args.json')
    args = Namespace(**utils.load_dict(filepath=args_fp))
    data = utils.load_dict(Path(config.DATA_DIR, 'train.json'))
    df = pd.DataFrame(data)
    
    artifacts = train.train(args, df)
    dp = config.MODEL_REGISTRY
    joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
    joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
    utils.save_dict(artifacts["performance"], Path(dp, "performance.json"))
    utils.save_dict(artifacts["args"], Path(dp, "args.json"))

@app.command()
def load_artifacts():
    '''
    Load artifacts from local storage.

    Returns:
        Dict : artifacts
    '''
    if len(os.listdir(config.MODEL_REGISTRY)) == 0:
        train_model()

    # Load objects from directory
    args = Namespace(**utils.load_dict(filepath=Path(config.MODEL_REGISTRY, 'args.json')))
    vectorizer = joblib.load(Path(config.MODEL_REGISTRY, 'vectorizer.pkl'))
    model = joblib.load(Path(config.MODEL_REGISTRY, 'model.pkl'))
    performance = utils.load_dict(filepath=Path(config.MODEL_REGISTRY, 'performance.json'))

    return {
        "args": args,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


@app.command()
def predict_news(text : List = []):
    ''' Makes prediction on a news text'''
    artifacts = load_artifacts()
    predictions = predict.predict(text, artifacts)
    logger.info(predictions)