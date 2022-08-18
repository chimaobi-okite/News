from typing import Dict
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def evaluate(y_true: np.ndarray, y_pred : np.ndarray) -> Dict:
    """Performance metrics using ground truths and predictions.
    Args:
        y_true (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.
    Returns:
        Dict: performance metrics.
    """

    metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    performance = {'precision': metrics[0], 'recall': metrics[1], 'f1': metrics[2]}
    return performance