"""Load, clean and split the Default of Credit Card Clients dataset (UCI 350)."""
from pathlib import Path # Path = a file path object, lets us join folders with "/" 

import pandas as pd
from sklearn.model_selection import train_test_split # scikit-learn function that cuts the data into train / test

from credit_default.config import ROOT, load_config  # ROOT = project folder, load_config = reads config.yaml

class DatasetLoader:
    """Load the credit default data, clean it and split it once into train / test."""

    def __init__(self, config: dict | None = None) -> None:
        """Read the file path, target, test size and random state from the config."""
        cfg = config or load_config()  # use the config given (e.g. a fake one in tests), otherwise read config.yaml
        self.path: Path = ROOT / cfg["paths"]["raw_data"]   # full path to the data file = project folder + relative path from the config
        self.target: str = cfg["target"]  # name of the column we want to predict ("default")
        self.test_size: float = cfg["test_size"] # share of rows kept aside for the final test (0.2 = 20%)
        self.random_state: int = cfg["random_state"] # fixed seed (42) so the random split is the same every run

    def load_data(self) -> pd.DataFrame: 
        """loads the UCI file from data -> raw and if its missing it downloads it with ucimlrepo"""
        try:  # try the normal way first: read the file already in the repo
            df = pd.read_excel(self.path, header=1)    # header=1: the file has 2 header rows, the real column names are in the 2nd one
        except FileNotFoundError:  # file not found -> download it instead
            from ucimlrepo import fetch_ucirepo   # imported here only, since we only need it when the file is missing
            df = fetch_ucirepo(id=350).data.original     #go to the UCI repo and get dataset num 350 (credit card dataset), return bundle with data, take the part that holds data, gives the whole table in one piece, and stores it in df   
        return df.rename(columns={"default payment next month": self.target}) #renaming it to make it easier to type 

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a copy, replace the undocumented values in EDUCATION and MARRIAGE."""
        df = df.copy()   # never modify the original
        df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)  #replaces 0,5,6 are sentinel values so replace with 4 which means "others"                                    
        df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3) #0 is also undocumented for this column
        return df

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split once into train / test, stratified. Returns X_train, X_test, y_train, y_test."""
        X = df.drop(columns=self.target)    # X = the clues: every column except the answer
        y = df[self.target]                  # y = the answer to guess: 1 = defaulted, 0 = did not
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state)  # cut the rows into 80% train / 20% test, # stratify=y keeps ~22% defaulters in both sets, random_state makes it repeatable
        return X_train, X_test, y_train, y_test      # same order as scikit-learn: inputs first, then answers



 
