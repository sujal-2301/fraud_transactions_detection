import streamlit as st
import pandas as pd
import joblib

# Paths and constants
MODEL_PATH = "models/rf_fraud_model.pkl"
DATA_PATH = "data/credit_card_transactions.csv"
THRESHOLD = 0.28


@st.cache_data
def load_model():
    """Load the trained Random Forest model from disk."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_mappings():
    """Recreate category-to-code mappings from the original dataset."""
    df = pd.read_csv(DATA_PATH)
    # Drop auto-index column if present
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    # Parse dates and compute age
    df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob_dt"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_dt"] - df["dob_dt"]).dt.days // 365

    categoricals = ["merchant", "category", "gender", "job", "state"]
    mappings = {}
    for col in categoricals:
        mappings[col] = list(df[col].astype("category").cat.categories)
    return mappings


def encode_row(raw: dict, mappings: dict) -> dict:
    """Encode a single input row's categorical fields using stored mappings."""
    encoded = {}
    for col, cats in mappings.items():
        val = raw[col]
        # Unseen categories map to -1
        encoded[col] = cats.index(val) if val in cats else -1
    return encoded


def main():
    st.title("💳 Fraud Detection Demo")
    st.write(f"Threshold for fraud flagging: {THRESHOLD}")

    # Load model and category mappings
    model = load_model()
    mappings = load_mappings()

    st.sidebar.header("Manual Input")
    # Build input form in sidebar
    raw_input = {}
    for col, cats in mappings.items():
        raw_input[col] = st.sidebar.selectbox(col, cats)
    # Numeric inputs
    raw_input["amt"] = st.sidebar.number_input("amt", value=0.0)
    raw_input["city_pop"] = st.sidebar.number_input("city_pop", value=0.0)
    raw_input["merch_lat"] = st.sidebar.number_input("merch_lat", value=0.0)
    raw_input["merch_long"] = st.sidebar.number_input("merch_long", value=0.0)
    raw_input["age"] = st.sidebar.number_input("age", value=0)

    # Encode and prepare DataFrame
    enc = encode_row(raw_input, mappings)
    X_single = pd.DataFrame([{**enc,
                              "amt": raw_input["amt"],
                              "city_pop": raw_input["city_pop"],
                              "merch_lat": raw_input["merch_lat"],
                              "merch_long": raw_input["merch_long"],
                              "age": raw_input["age"]}])

    # Make predictions
    prob = model.predict_proba(X_single)[0, 1]
    pred = int(prob > THRESHOLD)

    st.subheader("🔍 Manual Prediction")
    st.write("🚨 Fraud" if pred else "✅ Legit")
    st.write(f"Fraud probability: {prob:.2%}")

    # Batch predictions via CSV
    st.header("📁 Batch Predictions (CSV)")
    uploaded = st.file_uploader(
        "Upload a CSV file with transactions", type="csv")
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        # Drop auto-index if present
        if "Unnamed: 0" in df_batch.columns:
            df_batch.drop("Unnamed: 0", axis=1, inplace=True)
        # Compute age
        df_batch["trans_dt"] = pd.to_datetime(
            df_batch["trans_date_trans_time"])
        df_batch["dob_dt"] = pd.to_datetime(df_batch["dob"])
        df_batch["age"] = (df_batch["trans_dt"] -
                           df_batch["dob_dt"]).dt.days // 365
        # Encode categoricals
        for col, cats in mappings.items():
            df_batch[col] = df_batch[col].apply(
                lambda x: cats.index(x) if x in cats else -1)
        # Prepare feature matrix
        feature_cols = list(mappings.keys()) + \
            ["amt", "city_pop", "merch_lat", "merch_long", "age"]
        X_batch = df_batch[feature_cols]
        # Predict
        df_batch["fraud_prob"] = model.predict_proba(X_batch)[:, 1]
        df_batch["fraud_flag"] = (
            df_batch["fraud_prob"] > THRESHOLD).astype(int)
        st.subheader("Batch Prediction Results")
        st.dataframe(df_batch)


if __name__ == "__main__":
    main()
