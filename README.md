# 📡 Telecom Customer Churn Predictor

An end-to-end machine learning project that predicts whether a telecom customer is likely to cancel their service, with an interactive Streamlit app for exploring individual customers.

The model is trained on the public **IBM Telco Customer Churn** dataset (7,043 customers, about 26.5% churn).

## Results

<!-- RESULTS:START -->
All numbers are measured on a **held-out test set of 1,407 customers** that was never used for tuning or threshold selection.

| Model | ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|---|
| Baseline (always predict "stay") | 0.500 | 0.000 | n/a | 0.000 | 0.734 |
| Logistic Regression | 0.834 | 0.791 | 0.488 | 0.604 | 0.724 |
| **XGBoost (tuned)** ← selected | **0.840** | 0.746 | 0.542 | **0.628** | 0.765 |

- The selected model catches about **75% of customers who actually churn**, and about **54% of the customers it flags really do leave**.
- Train ROC-AUC is 0.863 against test 0.840, so overfitting is small. (An untuned XGBoost with default settings scored 0.989 on train and 0.810 on test.)
- Accuracy is a weak metric here because 73% of customers don't churn, so ROC-AUC, recall and precision matter more.
- The selected model uses the tuned cutoff of 0.58; the other model is scored at the default 0.5. ROC-AUC does not depend on the cutoff, so it is the fairest comparison between the two.
<!-- RESULTS:END -->

Figures are in [`reports/`](reports/): ROC curve, confusion matrix and feature importance. The full comparison table is in `reports/metrics.json`.

## How it works

1. **Cleaning** – `TotalCharges` contains blank strings for brand-new customers; those 11 rows are dropped.
2. **Feature engineering** – customers are grouped into tenure bands (`0-1 yr` … `5+ yrs`). The logic lives in `churn_features.py` and is shared by training and the app, so they cannot drift apart.
3. **Preprocessing** – standard scaling for numeric features, one-hot encoding for categorical features, wrapped in a scikit-learn `Pipeline` together with the classifier.
4. **Model selection** – all choices are made on the training set only:
   - XGBoost is tuned with 40 random hyperparameter configurations × 5-fold cross-validation (shallow trees, regularization, `scale_pos_weight` set to the real class ratio).
   - It is compared against Logistic Regression and a majority-class baseline using cross-validated ROC-AUC.
5. **Decision threshold** – chosen to maximize F1 on out-of-fold training predictions (the chosen value is saved in `models/model_metadata.json`), rather than using the default 0.5.
6. **One-time test evaluation** – the test set is used once, at the end, to report the numbers above.

### What drives churn in this data

The strongest signals in the model are month-to-month contracts (by far the largest), fiber-optic internet, no online security, no tech support, paying by electronic check, and short tenure (under a year). Feature importance shows what the model relies on, not what *causes* churn.

## Run it

Tested on Python 3.13.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset into data/ (see data/README.md)

# 3. (Optional) retrain the model and regenerate reports - takes under a minute
python train_model.py

# 4. Launch the app
streamlit run app.py
```

A trained model is already included in `models/`, so step 3 is optional. If you use different library versions than those in `requirements.txt` and the model fails to load, run `python train_model.py` to rebuild it for your environment.

### Troubleshooting

**`XGBoostError: input stream corrupted` or "The saved model could not be loaded"**

Saved XGBoost/scikit-learn models only load with the same library versions that created them. Rebuild the model for your own environment:

1. Put the dataset in `data/` (see `data/README.md`).
2. Run `python train_model.py` (under a minute).
3. Run `streamlit run app.py` again.

You can check your installed versions with `pip list` and compare them with `trained_with` in `models/model_metadata.json`.

### Run the tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests check the feature engineering, model sanity (including a quality floor and an overfitting guard) and the app's behaviour end to end.

## The app

Enter a customer's demographics, services and billing details to get a churn probability and a risk band:

<!-- BANDS:START -->
| Band | Meaning |
|---|---|
| 🟢 Low | probability below 35% |
| 🟡 Medium | 35% to 58% |
| 🔴 High | 58% or above (the tuned decision threshold) |
<!-- BANDS:END -->

Inputs depend on each other the way the real data does: customers with no internet service have no add-ons, and customers with no phone service have no multiple-lines option. Total Charges defaults to tenure × monthly charges.

## Project structure

```
├── app.py                  # Streamlit app
├── train_model.py          # Training, tuning, comparison, evaluation, reports
├── churn_features.py       # Shared cleaning and feature engineering
├── requirements.txt        # Pinned dependencies
├── requirements-dev.txt    # + pytest
├── data/                   # Put the dataset here (not committed)
├── models/
│   ├── churn_pipeline.pkl  # Trained pipeline
│   └── model_metadata.json # Chosen model, threshold, test metrics
├── reports/                # Metrics JSON and figures
└── tests/test_churn.py     # Pytest suite
```

## Limitations

- The dataset is a single snapshot from one (fictional) provider, so results may not carry over to other markets or time periods.
- Roughly half of the customers flagged as high risk will not actually churn. The model is best used to prioritize retention outreach, not to make automatic decisions about customers.
- Recommendations shown in the app are generic suggestions, not outputs of a causal model.

## Dataset

Telco Customer Churn, IBM sample data: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
