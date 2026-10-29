"""Tests for the pipeline, the predictions and the threshold: same checks as the course "✅ Check" cells."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

from credit_default.config import load_config
from credit_default.data import DatasetLoader
from credit_default.evaluate import apply_thresholds, choose_threshold, evaluate
from credit_default.features import NUM
from credit_default.models import Trainer

loader, trainer = DatasetLoader(), Trainer()
df = loader.clean(loader.load_data())
X_train, X_test, y_train, y_test = loader.split(df)
model = trainer.build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train) #the fitted logistic regression, used by the leakage, probability and save/load tests. It’s fitted once here, so the tests stay fast.
p_train = model.predict_proba(X_train)[:, 1] #holds the training probabilities, used by both threshold tests.


def test_default_rate():  #s7 notebook 
    """About 22% of the clients default."""
    assert 0.2 < df[loader.target].mean() < 0.25, df[loader.target].mean() #0.2-0.25 range comes from the project brief (Final_project_ML_x_Python.pdf, Datasets table: credit default has “about 22% positives”).
 
 
def test_split_sizes(): #s1 notebook, asks: “is train really 80% of all the rows?”
    """The train share matches the config (80 / 20)."""
    assert abs(len(X_train) / len(df) - (1 - loader.test_size)) < 0.01 #the gap has to be smaller than 1% because an exact match is too strict for this 
 #Actual: train rows ÷ all rows = 0.8 , Wanted: 80% of rows in train, so 0.8
 
def test_split_same_rate(): #second assert S1 notebook 
    """Stratify: same share of defaulters in train and test."""
    assert abs(y_train.mean() - y_test.mean()) < 0.01 #we have no validation set, so it’s just train vs test.

def test_no_leakage():
    """The scaler learned its means from the training set only (S1 Part 5)."""
    scaler = model.named_steps["prep"].named_transformers_["num"]
    assert np.allclose(scaler.mean_, X_train[NUM].mean())

def test_proba_shape_and_range():
    """One probability pair per client, all between 0 and 1."""
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2), proba.shape
    assert ((proba >= 0) & (proba <= 1)).all()
 
 
def test_apply_thresholds_gives_0_or_1():
    """Decisions are only 0 or 1 (S4 2.3 check)."""
    assert set(np.unique(apply_thresholds(p_train, 0.5))) <= {0, 1}
 
 
def test_choose_threshold_reaches_recall():
    """The chosen threshold reaches the recall goal and goes below 0.5 (S5 4.2 check)."""
    threshold = choose_threshold(y_train, p_train, min_recall=0.60)
    assert 0 < threshold < 0.5, threshold
    assert recall_score(y_train, apply_thresholds(p_train, threshold)) >= 0.60
 
 
def test_baseline_finds_no_defaulter():
    """The baseline always says "no default", so its recall is 0 (S1 Part 7)."""
    baseline = trainer.build_baseline().fit(X_train, y_train)
    assert evaluate(baseline, X_test, y_test)["recall"] == 0
 
 
def test_save_and_load(tmp_path):
    """A saved then loaded model gives the same predictions."""
    cfg = load_config()
    cfg["paths"]["model"] = str(tmp_path / "model.joblib")  # temporary folder, so models/ is not touched
    t = Trainer(cfg)
    t.save(model)
    assert np.allclose(t.load().predict_proba(X_test), model.predict_proba(X_test))
