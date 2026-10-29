"""Train, save and load the credit default pipeline."""
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

#BaseEstimator is the parent class of every scikit-learn model (LogisticRegression, KNeighborsClassifier, DummyClassifier) 
from credit_default.config import ROOT, load_config
from credit_default.features import build_preprocessor  # imports the function from features.py

MODELS = {"logistic_regression": LogisticRegression, "knn": KNeighborsClassifier}  # names used in config.yaml


class Trainer:
    """Build the pipeline (preparation + model), save it and load it back."""

    def __init__(self, config: dict | None = None) -> None:
        """Read the model path, the random state and the chosen model from the config."""
        cfg = config or load_config()  # use the config given (e.g. a fake one in tests), otherwise read config.yaml
        self.model_path: Path = ROOT / cfg["paths"]["model"]  # where the fitted pipeline is saved (models/model.joblib)
        self.random_state: int = cfg["random_state"]  # same fixed seed as the split, so models are reproducible
        self.model_cfg: dict = cfg["model"]  # final model and its settings, chosen in the notebook

    def build_pipeline(self, classifier: BaseEstimator) -> Pipeline:
        """Put the preparation and a classifier in one unfitted Pipeline."""
        return Pipeline([("prep", build_preprocessor()), ("clf", classifier)])  # S7 3.1: same preparation for every model

    def build_baseline(self) -> Pipeline:
        """Pipeline that always predicts the majority class ("no default")."""
        return self.build_pipeline(DummyClassifier(strategy="most_frequent"))  # S1 Part 7: any real model must beat this

    def build_final(self) -> Pipeline:
        """Pipeline with the model and settings chosen in the notebook (read from config.yaml)."""
        clf = MODELS[self.model_cfg["name"]](**self.model_cfg["params"])  # e.g. LogisticRegression(C=0.1, max_iter=1000)
        return self.build_pipeline(clf)

    def save(self, model: Pipeline) -> None:  # train once, then reuse the model without retraining
        """Save the fitted pipeline to the path from the config."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)  # create models/ if it doesn't exist yet
        joblib.dump(model, self.model_path)  # write the whole pipeline (scaler + model) to models/model.joblib

    def load(self) -> Pipeline:
        """Load the fitted pipeline saved by save()."""
        return joblib.load(self.model_path)  # read it back exactly as it was saved
