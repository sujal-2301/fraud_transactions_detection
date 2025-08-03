import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# Paths and constants
MODEL_PATH = "models/rf_fraud_model.pkl"
DATA_PATH = "data/credit_card_transactions.csv"
THRESHOLD = 0.28

# Example transaction data for quick testing
EXAMPLE_TRANSACTIONS = {
    "Legitimate Transaction 1": {
        "merchant": "fraud_Rippin, Kub and Mann",
        "category": "misc_net",
        "gender": "F",
        "job": "Psychologist, counselling",
        "state": "NC",
        "amt": 4.97,
        "city_pop": 3495,
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
        "age": 30
    },
    "Legitimate Transaction 2": {
        "merchant": "fraud_Heller, Gutmann and Zieme",
        "category": "grocery_pos",
        "gender": "F",
        "job": "Special educational needs teacher",
        "state": "WA",
        "amt": 107.23,
        "city_pop": 149,
        "merch_lat": 49.159046999999994,
        "merch_long": -118.186462,
        "age": 40
    },
    "Fraudulent Transaction 1": {
        "merchant": "fraud_Rutherford-Mertz",
        "category": "grocery_pos",
        "gender": "M",
        "job": "Soil scientist",
        "state": "NC",
        "amt": 281.06,
        "city_pop": 885,
        "merch_lat": 36.430124,
        "merch_long": -81.17948299999999,
        "age": 30
    },
    "Fraudulent Transaction 2": {
        "merchant": "fraud_Goodwin-Nitzsche",
        "category": "grocery_pos",
        "gender": "F",
        "job": "Horticultural consultant",
        "state": "TX",
        "amt": 276.31,
        "city_pop": 1595797,
        "merch_lat": 29.273085,
        "merch_long": -98.83636,
        "age": 35
    },
    "Fraudulent Transaction 3": {
        "merchant": "fraud_Medhurst PLC",
        "category": "shopping_net",
        "gender": "M",
        "job": "Soil scientist",
        "state": "NC",
        "amt": 844.8,
        "city_pop": 885,
        "merch_lat": 35.987802,
        "merch_long": -81.25433199999999,
        "age": 28
    },
    "Legitimate Transaction 3": {
        "merchant": "fraud_Lind-Buckridge",
        "category": "entertainment",
        "gender": "M",
        "job": "Nature conservation officer",
        "state": "ID",
        "amt": 220.11,
        "city_pop": 4154,
        "merch_lat": 43.150704,
        "merch_long": -112.154481,
        "age": 57
    }
}


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


def display_prediction_result(prob, threshold):
    """Display prediction result with visual indicators."""
    pred = int(prob > threshold)

    # Create columns for better layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if pred:
            st.error("🚨 FRAUD DETECTED")
            st.markdown(f"**Fraud Probability: {prob:.1%}**")
            st.progress(prob)
        else:
            st.success("✅ LEGITIMATE TRANSACTION")
            st.markdown(f"**Fraud Probability: {prob:.1%}**")
            st.progress(prob)

    # Risk level indicator
    if prob < 0.1:
        risk_level = "🟢 Low Risk"
    elif prob < 0.3:
        risk_level = "🟡 Medium Risk"
    elif prob < 0.6:
        risk_level = "🟠 High Risk"
    else:
        risk_level = "🔴 Very High Risk"

    st.info(f"**Risk Level:** {risk_level}")


def main():
    # Page configuration
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main header
    st.title("💳 Fraud Transaction Detection System")
    st.markdown("---")

    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")

    # Threshold adjustment
    threshold = st.sidebar.slider(
        "Fraud Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=THRESHOLD,
        step=0.01,
        help="Adjust the sensitivity of fraud detection. Higher values = more strict."
    )

    st.sidebar.markdown("---")

    # Load model and mappings
    with st.spinner("Loading model and data..."):
        model = load_model()
        mappings = load_mappings()

    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎯 Quick Examples", "📝 Manual Input", "📁 Batch Upload"])

    with tab1:
        st.header("🚀 Quick Examples")
        st.markdown(
            "Try these example transactions to see how the model works:")

        # Example selection
        example_choice = st.selectbox(
            "Choose an example transaction:",
            list(EXAMPLE_TRANSACTIONS.keys()),
            help="Select an example to see how the model predicts fraud"
        )

        if example_choice:
            example_data = EXAMPLE_TRANSACTIONS[example_choice]

            # Display example details
            st.subheader(f"📋 Example: {example_choice}")

            # Create a nice display of the example data
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Transaction Details:**")
                st.write(f"💰 Amount: ${example_data['amt']:,.2f}")
                st.write(f"🏪 Merchant: {example_data['merchant']}")
                st.write(f"📂 Category: {example_data['category']}")

            with col2:
                st.markdown("**Customer Details:**")
                st.write(f"👤 Gender: {example_data['gender']}")
                st.write(f"💼 Job: {example_data['job']}")
                st.write(f"📍 State: {example_data['state']}")
                st.write(f"🎂 Age: {example_data['age']} years")

            # Make prediction
            enc = encode_row(example_data, mappings)
            X_example = pd.DataFrame([{**enc,
                                      "amt": example_data["amt"],
                                      "city_pop": example_data["city_pop"],
                                      "merch_lat": example_data["merch_lat"],
                                      "merch_long": example_data["merch_long"],
                                      "age": example_data["age"]}])

            prob = model.predict_proba(X_example)[0, 1]

            st.markdown("---")
            st.subheader("🔍 Prediction Result")
            display_prediction_result(prob, threshold)

    with tab2:
        st.header("📝 Manual Transaction Input")
        st.markdown(
            "Enter transaction details manually to test fraud detection:")

        # Create a form for better UX
        with st.form("transaction_form"):
            st.subheader("Transaction Information")

            col1, col2 = st.columns(2)

            with col1:
                merchant = st.selectbox(
                    "🏪 Merchant", mappings["merchant"], help="Select the merchant where the transaction occurred")
                category = st.selectbox(
                    "📂 Category", mappings["category"], help="Select the transaction category")
                amount = st.number_input(
                    "💰 Amount ($)", min_value=0.01, value=50.0, step=0.01, help="Enter the transaction amount")
                city_pop = st.number_input(
                    "🏙️ City Population", min_value=0, value=100000, step=1000, help="Enter the city population")

            with col2:
                gender = st.selectbox(
                    "👤 Gender", mappings["gender"], help="Select customer gender")
                job = st.selectbox(
                    "💼 Job", mappings["job"], help="Select customer job")
                state = st.selectbox(
                    "📍 State", mappings["state"], help="Select customer state")
                age = st.number_input(
                    "🎂 Age", min_value=18, max_value=100, value=30, help="Enter customer age")

            st.subheader("Location Information")
            col3, col4 = st.columns(2)

            with col3:
                merch_lat = st.number_input("📍 Merchant Latitude", min_value=-90.0,
                                            max_value=90.0, value=40.7128, step=0.0001, help="Enter merchant latitude")

            with col4:
                merch_long = st.number_input("📍 Merchant Longitude", min_value=-180.0,
                                             max_value=180.0, value=-74.0060, step=0.0001, help="Enter merchant longitude")

            submitted = st.form_submit_button(
                "🔍 Predict Fraud", type="primary")

            if submitted:
                # Prepare input data
                raw_input = {
                    "merchant": merchant,
                    "category": category,
                    "gender": gender,
                    "job": job,
                    "state": state,
                    "amt": amount,
                    "city_pop": city_pop,
                    "merch_lat": merch_lat,
                    "merch_long": merch_long,
                    "age": age
                }

                # Encode and predict
                enc = encode_row(raw_input, mappings)
                X_single = pd.DataFrame([{**enc,
                                          "amt": raw_input["amt"],
                                          "city_pop": raw_input["city_pop"],
                                          "merch_lat": raw_input["merch_lat"],
                                          "merch_long": raw_input["merch_long"],
                                          "age": raw_input["age"]}])

                prob = model.predict_proba(X_single)[0, 1]

                st.markdown("---")
                st.subheader("🔍 Prediction Result")
                display_prediction_result(prob, threshold)

    with tab3:
        st.header("📁 Batch Transaction Analysis")
        st.markdown(
            "Upload a CSV file with multiple transactions for batch analysis:")

        uploaded = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with transaction data. Make sure it has the same columns as the training dataset."
        )

        if uploaded:
            try:
                df_batch = pd.read_csv(uploaded)

                # Show file info
                st.success(f"✅ File uploaded successfully! {len(df_batch)} transactions found.")

                # Drop auto-index if present
                if "Unnamed: 0" in df_batch.columns:
                    df_batch.drop("Unnamed: 0", axis=1, inplace=True)

                # Show sample of uploaded data
                st.subheader("📋 Sample of Uploaded Data")
                st.dataframe(df_batch.head(), use_container_width=True)

                # Process button
                if st.button("🔍 Analyze Transactions", type="primary"):
                    with st.spinner("Processing transactions..."):
                        # Compute age if date columns exist
                        if "trans_date_trans_time" in df_batch.columns and "dob" in df_batch.columns:
                            df_batch["trans_dt"] = pd.to_datetime(
                                df_batch["trans_date_trans_time"])
                            df_batch["dob_dt"] = pd.to_datetime(
                                df_batch["dob"])
                            df_batch["age"] = (
                                df_batch["trans_dt"] - df_batch["dob_dt"]).dt.days // 365

                        # Encode categoricals
                        for col, cats in mappings.items():
                            if col in df_batch.columns:
                                df_batch[col] = df_batch[col].apply(
                                    lambda x: cats.index(x) if x in cats else -1)

                        # Prepare feature matrix
                        feature_cols = list(
                            mappings.keys()) + ["amt", "city_pop", "merch_lat", "merch_long", "age"]
                        available_cols = [
                            col for col in feature_cols if col in df_batch.columns]

                        if len(available_cols) == len(feature_cols):
                            X_batch = df_batch[feature_cols]

                            # Predict
                            df_batch["fraud_prob"] = model.predict_proba(X_batch)[
                                :, 1]
                            df_batch["fraud_flag"] = (
                                df_batch["fraud_prob"] > threshold).astype(int)

                            # Display results
                            st.subheader("📊 Analysis Results")

                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Transactions", len(df_batch))
                            with col2:
                                st.metric("Fraudulent",
                                          df_batch["fraud_flag"].sum())
                            with col3:
                                st.metric("Legitimate", len(
                                    df_batch) - df_batch["fraud_flag"].sum())
                            with col4:
                                fraud_rate = df_batch["fraud_flag"].mean(
                                ) * 100
                                st.metric("Fraud Rate",
                                          f"{fraud_rate:.1f}%")

                            # Results table
                            st.subheader("📋 Detailed Results")
                            st.dataframe(
                                df_batch, use_container_width=True)

                            # Download results
                            csv = df_batch.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="fraud_analysis_results.csv",
                                mime="text/csv"
                            )
                        else:
                            missing_cols = set(feature_cols) - \
                                set(df_batch.columns)
                            st.error(
                                f"❌ Missing required columns: {', '.join(missing_cols)}")
                            st.info(
                                "Please ensure your CSV file contains all required columns.")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info(
                    "💡 **Tip**: Make sure your CSV file has the correct format with transaction data columns.")
        else:
            # Show sample data when no file is uploaded
            st.info("💡 **Upload a CSV file** to analyze your transaction data, or try the Quick Examples tab to see how the system works.")

            # Show example of expected CSV format
            with st.expander("📋 Expected CSV Format"):
                st.markdown("""
                Your CSV file should contain these columns:
                - `merchant`: Merchant name
                - `category`: Transaction category (e.g., grocery_pos, misc_net)
                - `amt`: Transaction amount
                - `gender`: Customer gender (F/M)
                - `job`: Customer job title
                - `state`: Customer state
                - `city_pop`: City population
                - `merch_lat`: Merchant latitude
                - `merch_long`: Merchant longitude
                - `age`: Customer age
                
                **Optional columns:**
                - `trans_date_trans_time`: Transaction timestamp
                - `dob`: Date of birth (for age calculation)
                """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>💡 <strong>Tip:</strong> Start with the Quick Examples tab to see how the model works!</p>
        <p>🔧 <strong>Model:</strong> Random Forest Classifier | <strong>Threshold:</strong> {threshold:.2f}</p>
        </div>
        """.format(threshold=threshold),
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
