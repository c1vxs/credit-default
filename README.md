# Credit card default

Predicts whether a credit card client will default on their payment next month (UCI dataset 350).


## 1. FRAMING THE PROBLEM: 

We are building a small tool that answers one pivotal question: "Will this client default on next month's payment?" 
Each month, the model chooses which clients need to be acted on early, for example to whom must we send a reminder call, a payment plan or a review of their credit limit before they actually default.

The users? The bank's credit risk/ collections team

The cost of each mistake: 

- False negative (the client defaults but is flagged as safe): 
  The bank does nothing and loses the unpaid balance. COST: they will lose money.

- False positive (the client is safe but is flagged as a defaulter): 
  The bank calls him unnecessarily, the client might be annoyed or have his limit lowered. COST: wasted efforts and also goodwill.  

If the model misses someone that defaults (FN), it costs more, so we care the most about recall (catching defaulters), but that is not the only thing we care about. If we flag everyone as defaulters, that means recall = 1 and precision ≈ 22%, so the team wastes almost all of its calls. We fix a recall goal and then get the best precision we can at that recall. 

## 2. Data
- Source: “Default of Credit Card Clients”, UCI Machine Learning Repository, id 350
- Licence: CC BY 4.0. 
- Size: 30,000 clients, 23 features, about 22% default
- Where: the .xls is included in data/raw/, and the code downloads it from UCI if it’s missing


## 3. Installation 


Requires Python 3.10 or later.

 ``` bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]" 
```

## 4. Usage

 ``` bash
credit-default train 
credit-default evaluate
credit-default predict --input new_case.json 
```


example output for evaluate: 

```text
TEST precision=0.692 recall=0.244 F1=0.361
TEST average precision=0.495 ROC-AUC=0.710
confusion matrix [[TN, FP], [FN, TP]]: [[4529, 144], [1003, 324]]
```

short note:

- the threshold comes from config.yaml
- new_case.json holds one client with the 23 feature columns, and you can copy it to make your own
- if the credit-default command isn’t found, use python -m credit_default.cli train (and so on). 

## 5. Tests

 ``` bash
pytest
ruff check .
```

The tests cover cleaning, split, no leakage, predictions, threshold, save and load.

## 6. Project layout 

```text
config.yaml              # paths, seed, test size, threshold
data/raw/                # the UCI .xls file
src/credit_default/
    config.py            # reads config.yaml
    data.py              # DatasetLoader: load, clean, split
    features.py          # ColumnTransformer
    models.py            # Trainer: pipelines, baseline, save, load
    evaluate.py          # threshold and metrics
    cli.py               # train, evaluate, predict
tests/                   # pytest
notebooks/analysis.ipynb # the modelling story, calls the package
``` 

## 7. Team and AI use

- Mahamadou CISSE
- Chris TAHAN 

Link to the report and slides:

AI USAGE: check AI_USAGE.md file.