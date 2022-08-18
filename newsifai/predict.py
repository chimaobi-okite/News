from typing import Dict, List
import numpy as np
from newsifai import utils
from config import config
from pathlib import Path


def predict(texts : List, artifacts : Dict) -> Dict:
    """Predict category for given text.
    Args:
        text (str): raw input text to classify.
        artifacts (Dict): artifacts from a run.
    Returns:
        Dict: predictions for input text.
    """
    cat2index = utils.load_dict(Path(config.DATA_DIR, 'catmapping.json'))
    index2cat = {v:k for k,v in cat2index.items()}
    x = artifacts['vectorizer'].transform(texts)
    y_probs = np.max(artifacts['model'].predict_proba(x), axis = 1)
    y_preds = artifacts['model'].predict(x)
    categories = np.vectorize(index2cat.get)(y_preds)
    predictions =  [{
            'input_text': texts[i],
            'predicted_category': categories[i],
            'predicted_prob' : y_probs[i]
        } for i in range(len(categories))]
        
    return predictions