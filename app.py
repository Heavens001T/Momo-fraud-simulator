import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Mobile Money Fraud Simulator", layout="centered")

st.title("📱 AI-Driven Mobile Money Fraud Detection System")
st.write("This system simulates mobile money transactions and uses AI to detect fraud.")

# -----------------------
# Generate synthetic data
# -----------------------
np.random.seed(42)

data = pd.DataFrame({
    "amount": np.random.randint(500, 500000, 500),
    "hour": np.random.randint(0, 24, 500),
    "transactions_last_hour": np.random.randint(1, 20, 500),
    "new_device": np.random.randint(0, 2, 500),
})

# Fraud rule (for training)
data["fraud"] = (
    (data["amount"] > 150000) &
    (data["transactions_last_hour"] > 5) &
    (data["new_device"] == 1)
).astype(int)

X = data.drop("fraud", axis=1)
y = data["fraud"]

model = RandomForestClassifier()
model.fit(X, y)

# -----------------------
# User input
# -----------------------
st.subheader("🔍 Simulate Transaction")

amount = st.slider("Transaction Amount (FCFA)", 500, 500000, 5000)
hour = st.slider("Hour of Transaction", 0, 23, 12)
tx_count = st.slider("Transactions in Last Hour", 1, 20, 1)
new_device = st.selectbox("New Device Used?", ["No", "Yes"])

new_device_val = 1 if new_device == "Yes" else 0

input_data = pd.DataFrame([[amount, hour, tx_count, new_device_val]],
                          columns=X.columns)

# -----------------------
# Prediction
# -----------------------
if st.button("Analyze Transaction"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🧠 AI Decision")

    st.write(f"**Fraud Risk Score:** {round(probability * 100, 2)}%")

    if probability < 0.3:
        st.success("✅ Transaction Approved")
    elif probability < 0.7:
        st.warning("⚠️ OTP Verification Required")
    else:
        st.error("❌ Transaction Blocked (High Fraud Risk)")
