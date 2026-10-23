import json
import os

import joblib
import pandas as pd
import streamlit as st

from churn_features import FEATURE_COLUMNS, add_tenure_group

MODEL_PATH = "models/churn_pipeline.pkl"
META_PATH = "models/model_metadata.json"

# 1. Page configuration
st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="wide")

# 2. Custom CSS
st.markdown("""
    <style>
    .main-header { font-size: 36px !important; font-weight: bold; color: #1E3A8A; text-align: center; }
    .sub-header { font-size: 18px !important; color: #6B7280; text-align: center; margin-bottom: 2rem; }
    div.stButton > button { background-color: #2563EB; color: white; font-size: 18px; font-weight: bold; height: 50px; border-radius: 8px; }
    div.stButton > button:hover { background-color: #1D4ED8; border-color: #1D4ED8; color: white; }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


metadata = load_metadata()

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python train_model.py` first to create it.")
    st.stop()

try:
    pipeline = load_model()
except Exception as exc:  # pickled models can fail to load across library versions
    import sklearn
    import xgboost

    trained = metadata.get("trained_with", {})
    st.error("⚠️ The saved model could not be loaded in this environment.")
    st.markdown(
        "Saved models only load with the **same library versions** they were trained with. "
        "The fix takes about a minute:\n\n"
        "1. Download the dataset into the `data/` folder (see `data/README.md`)\n"
        "2. Run `python train_model.py`\n"
        "3. Restart the app with `streamlit run app.py`"
    )
    st.code(
        f"Trained with : xgboost {trained.get('xgboost', '?')}, scikit-learn {trained.get('scikit-learn', '?')}\n"
        f"Installed now: xgboost {xgboost.__version__}, scikit-learn {sklearn.__version__}\n"
        f"Error        : {type(exc).__name__}: {exc}"
    )
    st.stop()
THRESHOLD = float(metadata.get("threshold", 0.5))
MEDIUM_FLOOR = round(THRESHOLD * 0.6, 2)  # below this = low risk; between = medium; >= THRESHOLD = high

# Header
st.markdown('<p class="main-header">📡 Customer Churn Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict the probability of a customer leaving the telecom service using Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# 3. Input form
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
    partner = st.selectbox("Has Partner?", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

with col2:
    st.subheader("📶 Service Details")
    # Dataset range is 1-72 months (brand-new customers with 0 months have no billing history)
    tenure = st.slider("Tenure (Months)", 1, 72, 12, help="How long has the customer been with the company?")
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    # Dependent fields: the model was trained on "No phone service" / "No internet service"
    # values, so these must be set automatically instead of left to the user.
    if phone_service == "No":
        multiple_lines = "No phone service"
        st.caption("Multiple Lines: *No phone service* (set automatically)")
    else:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

    with st.expander("View More Service Options"):
        add_on_names = {
            "OnlineSecurity": "Online Security",
            "OnlineBackup": "Online Backup",
            "DeviceProtection": "Device Protection",
            "TechSupport": "Tech Support",
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming Movies",
        }
        add_ons = {}
        if internet_service == "No":
            st.caption("No internet service, so these add-ons are not available.")
            for key in add_on_names:
                add_ons[key] = "No internet service"
        else:
            for key, label in add_on_names.items():
                add_ons[key] = st.selectbox(label, ["No", "Yes"])

with col3:
    st.subheader("💳 Billing & Contract")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=5.0)
    # Defaults to tenure x monthly charges so the two stay consistent; editable if the customer's history differs.
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=50.0,
        help="Defaults to Tenure x Monthly Charges. Edit it if you know the actual total.",
    )

# Build the model input in the exact column order used for training
input_data = {
    "gender": gender,
    "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    **add_ons,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}
input_df = add_tenure_group(pd.DataFrame([input_data]))[FEATURE_COLUMNS]

st.markdown("---")

# 4. Centered submit button
_, btn_col, _ = st.columns([1, 1, 1])
with btn_col:
    predict_button = st.button("🚀 Analyze Churn Risk", use_container_width=True)

# 5. Results
if predict_button:
    st.markdown("---")
    with st.spinner("Analyzing customer profile & running machine learning model..."):
        probability = float(pipeline.predict_proba(input_df)[0][1])

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(label="Churn Probability", value=f"{probability * 100:.1f}%")

    with res_col2:
        st.progress(min(max(int(round(probability * 100)), 0), 100))
        if probability >= THRESHOLD:
            st.error("🚨 **High Risk Customer:** This customer is likely to leave.")
            st.info("💡 **Recommendation:** Offer a discount, upgrade their tech support, or suggest a 1-year contract to retain them.")
        elif probability >= MEDIUM_FLOOR:
            st.warning("⚠️ **Medium Risk Customer:** Some churn signals are present, but not enough to flag as likely to leave.")
            st.info("💡 **Recommendation:** Monitor this customer and consider a proactive check-in or a loyalty offer.")
        else:
            st.success("✅ **Low Risk Customer:** This customer is likely to stay.")
            st.info("💡 **Recommendation:** Customer is satisfied. Good candidate for upselling premium services.")

# 6. Model info
if metadata:
    with st.expander("ℹ️ About this model"):
        tm = metadata.get("test_metrics", {})
        st.write(
            f"**Model:** {metadata.get('model')}  \n"
            f"**Trained on:** {metadata.get('n_train')} customers, evaluated on {metadata.get('n_test')} unseen customers  \n"
            f"**Risk bands:** Low < {MEDIUM_FLOOR:.0%} · Medium {MEDIUM_FLOOR:.0%}–{THRESHOLD:.0%} · High ≥ {THRESHOLD:.0%}"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.3f}")
        m2.metric("Recall", f"{tm.get('recall', 0):.1%}")
        m3.metric("Precision", f"{tm.get('precision', 0):.1%}")
        m4.metric("F1", f"{tm.get('f1', 0):.3f}")
        st.caption(
            f"Metrics are from a held-out test set. The model catches about {tm.get('recall', 0):.0%} of customers who "
            f"actually churn, and about {tm.get('precision', 0):.0%} of the customers it flags really do leave, "
            "so treat results as a prioritisation aid, not a certainty."
        )
