**Tool:** Claude (Anthropic), through claude.ai.

The brief allows generative AI as a helper if every use is declared. This file lists what we used it for, part by part. We read, ran and tested everything we submit, and we can explain every line of it.

## How we used it

- **Planning:** a timeline for the 13 days, the split of tasks between the two of us, and a step-by-step plan for the EDA.
- **Understanding:** explanations of the brief, the dataset, the methods (pipeline, cross-validation, out-of-fold threshold) and the whole project, in plain words.
- **Explaining errors and setup problems:** Windows blocking `.exe` launchers (we use `py -m ...` instead), venv and kernel setup in VS Code, Git merge conflicts, `ruff` errors, reading the `.xls` file (`xlrd`, `header=1`).
- **Pointing to the course:** for each piece of code, we asked which course notebook (S1 to S8) shows the pattern, then followed that style.
- **Reviewing our work:** finding bugs in our drafts, checking type hints and docstrings, a fresh-install test of the repository, and a checklist of what the brief still required.
- **Drafting some code and text:** listed below, with what we checked or changed.

Decisions stayed ours: the dataset, the problem framing (who uses the tool and the cost of each mistake), the cleaning rules, keeping `PAY_` numeric, the 60 % recall goal (suggested by AI from Lab 2, kept after checking the trade-off table) and the rule for picking the final model. AI explained the options.

## Christina Tahan (package, CLI, tests, README)

| Part | What AI did | What we did |
|---|---|---|
| Project setup (`pyproject.toml`, `config.yaml`, `.gitignore`, Git) | Gave the PowerShell and Git commands (selective `git add`, restoring a file before pulling), explained the fixes for setup errors | Chose the layout from the brief, ran every command, fixed the conflicts |
| `data.py` (`DatasetLoader`) | Reviewed our draft and pointed out bugs (a missing `return`, a typo in `fetch_ucirepo`, a missing docstring); explained `header=1` | Wrote the class; cleaning rules (EDUCATION 0/5/6 → 4, MARRIAGE 0 → 3) checked with `value_counts` in the notebook |
| `PAY_` codes | Analysed what −2 and 0 could mean and whether to recode them | Kept the columns numeric, since −2, −1 and 0 have the same low default rate |
| `config.py`, `features.py` | Helped with `load_config` and pointed to the `ColumnTransformer` pattern of S7 Part 3.1 | Chose the `NUM` / `CAT` lists |
| `models.py` (`Trainer`) | Helped write `save()` / `load()` (`joblib.dump` / `joblib.load`, `Path.mkdir(parents=True, exist_ok=True)`), which are not in the course notebooks; generated `build_final()`, which reads the model from `config.yaml` | Checked them with the save/load test and the CLI |
| `evaluate.py` | Helped turn the course's threshold method (S5 4.2, S7 Lab 2 protocol with `cross_val_predict`) into functions | Checked the threshold against the notebook (0.19) |
| `cli.py` | Helped with the `argparse` structure (subcommands `train`, `evaluate`, `predict`) | Ran the three commands on a fresh install |
| `tests/` | Drafted tests from the brief's list (cleaning, split, leakage, predictions, threshold, save/load) and helped fix failing ones | Ran them with `pytest`; all pass |
| Reproducibility | Traced the ROC-AUC difference between our machines (0.753 vs 0.754) to the SciPy version, then to rounding | Pinned `scipy==1.15.3` and added the note to the README |
| `README.md` | Reviewed the structure and the install / usage sections | Wrote the problem framing ourselves |
| Notebook (2 Nov) | Reviewed the notebook against the brief; wrote the cells for the ID check, the `PAY_` codes note, the "why 60 %" table and the error rates by sex | Placed them, ran the notebook and checked the numbers |
| Report and defense | Helped draft the problem paragraph and a defense preparation document (likely jury questions) | Rewrote them in our own words |
| `AI_USAGE.md` | Drafted this file from our chats | Checked every row |

## Mahamadou CISSE (EDA, modelling, notebook)

| Part | What AI did | What we did |
|---|---|---|
| Environment and Git | Gave the commands to clone the repo, create the venv (switching from Python 3.15 to 3.13), pin dependencies, and fix Git problems (rejected push, merge after pull, PowerShell vs cmd syntax) | Ran every command and checked the result |
| EDA and cleaning (Parts 1 to 3) | For each cell, proposed code based on the course notebook it came from (mainly S1), with a line-by-line explanation; drafted the short conclusions | Typed every cell myself, adapted the style to the course (no question-and-answer text), ran every check and compared the numbers. Once, the AI overwrote my typed cells directly; I restored my own version |
| Baseline and model comparison (Parts 4 and 5, first version) | Proposed the code from S6 solution 3.1 and Lab 2 B.1 (dummy model, `cross_validate` loop, `GridSearchCV`) and suggested average precision as the comparison metric | Typed and ran it, read the `best_params_` and filled in the tuned values |
| Recall goal | Suggested a 60 % recall goal, following Lab 2, and explained the out-of-fold threshold method | Kept 60 % with my teammate after checking the trade-off (50 to 80 % recall) |
| k-NN grid | Pointed out that the best k (51) was at the edge of the grid and listed the pros and cons of widening it | Decided with my teammate to widen it; re-ran the notebook; k = 201 chosen on cross-validation only; updated the numbers in the text of Parts 5, 7 and 8 |
| Notebook Parts 6 to 8 (29 Oct) | Rewritten by my teammate (with her AI) to call the package; my AI compared it with my version and checked that no cell uses the test set before Part 7 | Replaced my notebook with it, re-ran it and checked that the CLI gives the same results |
| Notebook (2 Nov) | Wrote the code of the precision/recall vs threshold plot and of the confusion matrix plot (`ConfusionMatrixDisplay`, not in the course notebooks) and the sentence under the threshold plot; re-computed the "why 60 %" table independently; pointed out the `refit=True` wording in Part 7 | Placed the cells, re-ran the whole notebook, checked that nothing changed (k = 201, threshold 0.19, test precision 0.436, recall 0.611) and fixed the wording |
| Reproducibility | Compared the library versions installed on our two machines (only SciPy differed) | Ran the CLI on my machine to confirm it matches the notebook (ROC-AUC 0.753) |
| Team communication | Drafted some of my messages to my teammate | Edited and sent them |
| Defense preparation | Built a revision guide in English (likely jury questions, quizzes) from the course materials and our code | Studied it and answered the questions myself |
| Report (ML parts) | TO FILL IN: what AI did for the report, e.g. "gave an outline with the key numbers and reviewed my draft" (delete this row if you did not use AI) | TO FILL IN: e.g. "wrote the text myself" |
| `AI_USAGE.md` | Drafted this section from my chats | Checked every row |
