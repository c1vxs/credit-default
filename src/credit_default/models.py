"""Train, save and load the credit default pipeline."""
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from credit_default.config import ROOT, load_config
from credit_default.features import build_preprocessor  # imports the function from features.py


class Trainer:
    """Build the pipeline (preparation + model), save it and load it back."""

    def __init__(self, config: dict | None = None) -> None:
        """Read the model path and the random state from the config."""
        cfg = config or load_config()  # use the config given (e.g. a fake one in tests), otherwise read config.yaml
        self.model_path: Path = ROOT / cfg["paths"]["model"]  # where the fitted pipeline is saved (models/model.joblib)
        self.random_state: int = cfg["random_state"]  # same fixed seed as the split, so models are reproducible

    def build_pipeline(self, classifier) -> Pipeline:
        """Put the preparation and a classifier in one unfitted Pipeline."""
        return Pipeline([("prep", build_preprocessor()), ("clf", classifier)]) # S7 3.1: same preparation for every model

    def build_baseline(self) -> Pipeline:
        """Pipeline that always predicts the majority class ("no default")."""
        return self.build_pipeline(DummyClassifier(strategy="most_frequent")) # S1 Part 7: any real model must beat this

    def save(self, model: Pipeline) -> None:#Training takes time, and you only want to do it once. After that, you want to use the trained model again and again without retraining.
        """Save the fitted pipeline to the path from the config."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)  # create models/ if it doesn't exist yet
        joblib.dump(model, self.model_path)  # write the whole pipeline (scaler + model) to models/model.joblib

    def load(self) -> Pipeline:
        """Load the fitted pipeline saved by save()."""
        return joblib.load(self.model_path)  # read it back exactly as it was saved
