"""Command-line tool: train, evaluate and predict."""
import argparse
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_default.config import load_config
from credit_default.data import DatasetLoader
from credit_default.evaluate import evaluate, apply_thresholds
from credit_default.models import Trainer


def train_model(cfg: dict) -> None:
    """Fit the model on the whole training set and save it."""
    loader = DatasetLoader(cfg) #same config as everywhere so path, target, test size, seed
    trainer = Trainer(cfg) #reads the model path and the seed from the same config
    X_train, X_test, y_train, y_test = loader.split(loader.clean(loader.load_data())) #stratified split
    model = trainer.build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train) #pipeline fitted on train only
    trainer.save(model) #joblib: train once then save so we can reuse in evaluate and predict
    print(f"Model trained on {len(X_train)} clients and saved to {trainer.model_path}")

def evaluate_model(cfg:dict) -> None:
    """Load the saved model and print its scores on the test set only one time"""
    loader = DatasetLoader(cfg)
    _, X_test, _, y_test = loader.split(loader.clean(loader.load_data())) #same seed and test set
    s = evaluate(Trainer(cfg).load(), X_test, y_test, threshold=cfg["threshold"]) #creates a trainer object which holds settings like model type, paths, hyperparameters and loads an already trained model. evaluate tests that model on the test set
    print(f"TEST precision={s['precision']:.3f} recall={s['recall']:.3f} F1={s['f1']:.3f}")
    print(f"TEST average precision={s['average_precision']:.3f} ROC-AUC={s['roc_auc']:.3f}")
    print("confusion matrix [[TN, FP], [FN, TP]]:", s["confusion_matrix"])

def predict_model(cfg: dict, input_path: str) -> None:
    """Load the saved model and predict one new client from a JSON file."""
    with open(input_path, encoding="utf-8") as f:  # open the JSON file the user gave with --input
        case = pd.DataFrame([json.load(f)])  # read the client, make it a 1-row table with the same columns as X_train
    proba = Trainer(cfg).load().predict_proba(case)[:, 1]  # load the saved pipeline, get the probability of default (S5 4.1)
    decision = apply_thresholds(proba, cfg["threshold"])[0]  # turn it into 0/1 with the threshold from config.yaml
    print(f"probability of default: {proba[0]:.3f}")  # e.g. 0.712
    print("decision:", "default" if decision == 1 else "no default")  # the answer in words


def main() -> None:
    """Read the command typed in the terminal and run it."""
    parser = argparse.ArgumentParser(description="Credit card default: train, evaluate, predict.")  # the tool itself
    sub = parser.add_subparsers(dest="command", required=True)  # it expects one command; its name goes in args.command
    sub.add_parser("train", help="fit the model and save it")  # command 1: no options
    sub.add_parser("evaluate", help="print the test metrics")  # command 2: no options
    predict = sub.add_parser("predict", help="predict one new client")  # command 3...
    predict.add_argument("--input", required=True, help="JSON file with one client")  # ...which needs a file
    args = parser.parse_args()  # read what was typed, e.g. "predict --input new_case.json"
    cfg = load_config()  # read config.yaml once, then pass it to the command
    if args.command == "train":  # run the function matching the command
        train_model(cfg)
    elif args.command == "evaluate":
        evaluate_model(cfg)
    else:  # the only command left is predict
        predict_model(cfg, args.input)  # args.input = the file path after --input


if __name__ == "__main__":  # only true when the file is run directly (py -m credit_default.cli ...)
    main()  # start the tool