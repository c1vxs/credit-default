"""Turns probabilities into decisions with a treshold and then scores those decisions"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


def out_of_fold_proba(model: Pipeline, X_train, y_train, random_state: int) -> np.ndarray:
    """Probability of default for each training client, predicted by a model that did not see them."""
    cv = StratifiedKFold(5, shuffle=True, random_state=random_state)  #seed from the config, not hard-coded
    return cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1] #out of fold probability
    #cross-validation gives every training client a probability predicted by a model that didn’t see them, which acts as a validation set

def apply_thresholds(proba, threshold: float=0.5) -> np.ndarray:
    """Return 1 where the probability of default is at least the threshold, else 0."""
    return (proba >= threshold).astype(int)  # S5 4.2: y_test_pred = (p_test >= threshold).astype(int)

def choose_threshold(y_true, proba: np.ndarray, min_recall: float) -> float:
    """Highest threshold whose recall is still at least min_recall (S5 4.2)."""
    thresholds = np.linspace(0.01, 0.99, 99)  # 0.01, 0.02, ..., 0.99
    recalls = [recall_score(y_true, apply_thresholds(proba, t)) for t in thresholds]  # recall for each threshold
    return max(t for t, r in zip(thresholds, recalls) if r >= min_recall)  # highest threshold still reaching the goal

def evaluate(model: Pipeline, X_test, y_test, threshold: float=0.5) -> dict: 
    """Score the fitted model once on the test set, with the chosen threshold."""
    proba = model.predict_proba(X_test)[:, 1]  # S5 4.1: probability of default for each test client
    y_pred = apply_thresholds(proba, threshold)  # S5 4.2: turn probabilities into 0/1 decisions
    return {
        "precision": precision_score(y_test, y_pred, zero_division=0),  # of the clients we flag, how many really default?
        "recall": recall_score(y_test, y_pred),  # of the real defaulters, how many do we find?
        "f1": f1_score(y_test, y_pred),  # balance of precision and recall
        "average_precision": average_precision_score(y_test, proba),  # area under the PR curve, on probabilities
        "roc_auc": roc_auc_score(y_test, proba),  # ranking quality over all thresholds, on probabilities
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # [[TN, FP], [FN, TP]]
    }

