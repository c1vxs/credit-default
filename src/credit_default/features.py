"""Feature preparation for the credit default model: scaling and encoding.

Each column is transformed according to what it represents:
- numeric columns (amounts, age, payment delays) are standardised with
  StandardScaler, so that large values like LIMIT_BAL don't dominate
  small ones like AGE;
- code columns (SEX, EDUCATION, MARRIAGE) are one-hot encoded, because
  their values are labels, not quantities.

The preprocessor is only built here, never fitted. It is fitted later inside
the Pipeline, on the training set only, so no information from the test set
leaks into the preparation.
"""
# "Why only these?” → This file only builds the preparation. The model and the pipeline go in models.py.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT = ["SEX","EDUCATION","MARRIAGE"] #categorical values, theyre codes not quantities, you cant say that EDUCATION = 4 is 2 times EDUCATION = 2 
NUM = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
#numerical values. he values are ordered levels of lateness, from −2 to 8, so the order means something

def build_preprocessor() -> ColumnTransformer: #Different columns need different treatment: scale the numbers, one-hot the codes.
    """Return the preparation: standardise the numbers, one-hot encode the codes."""
    prep = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    # numeric columns: rescale each to mean 0 and std 1, so big values like LIMIT_BAL don't dominate small ones like AGE
    # code columns: one 0/1 column per category; "ignore" avoids a crash on a category never seen in training
    #standard scaler because LIMIT_BAL is in the hundreds of thousands and AGE in the tens. Without scaling, models like logistic regression or k-NN are dominated by the big numbers.
    return prep 

#No .fit() in this file.
#Why: it gets fitted later, inside the pipeline, on the training set only.
#Where to look: S1_solution, Part 5: fit_prepare(X_train) learns “from the training set only”.
#“How do you avoid leakage?” → The preparation is never fitted in features.py. It goes into the Pipeline, and .fit(X_train, y_train) fits it on train only. The test set never influences the scaler’s means.