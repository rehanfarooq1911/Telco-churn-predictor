"""Regression tests for the bugs found in the original project plus basic model sanity checks.

Run from the project root:  pytest -q
"""
import json
import os
import sys

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # app.py loads the model via a relative path

from churn_features import FEATURE_COLUMNS, add_tenure_group  # noqa: E402

APP = os.path.join(ROOT, "app.py")


def widget(at, label):
    for w in list(at.selectbox) + list(at.slider) + list(at.number_input):
        if w.label == label:
            return w
    raise KeyError(label)


def has_widget(at, label):
    return any(w.label == label for w in list(at.selectbox) + list(at.slider) + list(at.number_input))


def analyze(at):
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return float(at.metric[0].value.rstrip("%")) / 100


@pytest.fixture
def app():
    return AppTest.from_file(APP, default_timeout=60).run()


# ---------------------------------------------------------------- features
def test_tenure_zero_is_not_missing():
    out = add_tenure_group(pd.DataFrame({"tenure": [0, 1, 12, 13, 72]}))
    assert out["TenureGroup"].tolist() == ["0-1 yr", "0-1 yr", "0-1 yr", "1-2 yrs", "5+ yrs"]
    assert "nan" not in out["TenureGroup"].tolist()


# ------------------------------------------------------------------- model
def test_model_loads_and_predicts_valid_probabilities():
    model = joblib.load("models/churn_pipeline.pkl")
    row = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 5,
        "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0, "TotalCharges": 425.0,
    }])
    p = model.predict_proba(add_tenure_group(row)[FEATURE_COLUMNS])[0][1]
    assert 0.0 <= p <= 1.0


def test_model_quality_floor():
    meta = json.load(open("models/model_metadata.json"))
    assert meta["test_metrics"]["roc_auc"] >= 0.82
    assert meta["test_metrics"]["recall"] >= 0.70
    # guard against the heavy overfitting of the original model (train 0.99 vs test 0.81)
    assert meta["train_roc_auc"] - meta["test_metrics"]["roc_auc"] < 0.06


# --------------------------------------------------------------------- app
def test_app_runs_without_errors(app):
    assert not app.exception


def test_no_internet_hides_addons_and_uses_correct_category(app):
    widget(app, "Internet Service").select("No").run()
    for label in ["Online Security", "Online Backup", "Device Protection",
                  "Tech Support", "Streaming TV", "Streaming Movies"]:
        assert not has_widget(app, label), f"{label} should be hidden when there is no internet"
    p_no_internet = analyze(app)
    assert 0.0 <= p_no_internet <= 1.0


def test_no_internet_customer_is_not_flagged_high_risk_like_before(app):
    """Originally, Internet='No' with add-ons left at 'No' produced 60.2% risk."""
    widget(app, "Internet Service").select("No").run()
    assert analyze(app) < 0.50


def test_no_phone_service_sets_multiple_lines_automatically(app):
    widget(app, "Phone Service").select("No").run()
    assert not has_widget(app, "Multiple Lines")
    assert not app.exception
    analyze(app)


def test_tenure_slider_minimum_matches_training_data(app):
    assert widget(app, "Tenure (Months)").min == 1


def test_risk_bands_high_medium_low(app):
    # Strong churn profile
    widget(app, "Contract Type").select("Month-to-month")
    widget(app, "Internet Service").select("Fiber optic")
    widget(app, "Payment Method").select("Electronic check")
    widget(app, "Tenure (Months)").set_value(2)
    widget(app, "Monthly Charges ($)").set_value(90.0).run()
    high = analyze(app)
    assert high > 0.6
    assert len(app.error) == 1

    # Strong loyalty profile
    app = AppTest.from_file(APP, default_timeout=60).run()
    widget(app, "Contract Type").select("Two year")
    widget(app, "Internet Service").select("DSL")
    widget(app, "Payment Method").select("Credit card (automatic)")
    widget(app, "Tenure (Months)").set_value(60)
    widget(app, "Monthly Charges ($)").set_value(50.0).run()
    low = analyze(app)
    assert low < 0.1
    assert len(app.success) == 1 and len(app.error) == 0


def test_total_charges_defaults_follow_tenure_and_monthly(app):
    widget(app, "Tenure (Months)").set_value(24).run()
    widget(app, "Monthly Charges ($)").set_value(70.0).run()
    assert widget(app, "Total Charges ($)").value == pytest.approx(24 * 70.0)


def test_corrupt_model_shows_friendly_message_instead_of_crashing():
    """Reproduces 'XGBoostError: input stream corrupted' seen on a user's machine."""
    import shutil
    import streamlit as st

    path = "models/churn_pipeline.pkl"
    shutil.copy(path, path + ".bak")
    try:
        with open(path, "wb") as f:
            f.write(b"this is not a valid model file")
        st.cache_resource.clear()
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception
        assert any("could not be loaded" in e.value for e in at.error)
        assert not at.button  # the form is not rendered when the model is unusable
    finally:
        shutil.move(path + ".bak", path)
        st.cache_resource.clear()
