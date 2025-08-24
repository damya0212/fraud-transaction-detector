
# Fraud Detection Streamlit App with RL Agent (Gymnasium-Compatible)

# Requirements (already assumed: pandas, sklearn)
# Run this in terminal first if needed:
# pip install streamlit stable-baselines3 gymnasium

import streamlit as st
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler
import time
import os

# Step 1: Generate synthetic transaction dataset
def generate_data(n_samples=3000):
    np.random.seed(42)
    data = pd.DataFrame({
        'amount': np.random.normal(100, 50, n_samples),
        'time': np.random.randint(0, 24, n_samples),
        'location_score': np.random.uniform(0, 1, n_samples),
        'device_score': np.random.uniform(0, 1, n_samples),
        'user_age': np.random.randint(18, 70, n_samples),
        'txn_type': np.random.choice(['crypto', 'merchant', 'international'], n_samples),
        'country': np.random.choice(['USA', 'India', 'Germany', 'Iran', 'North Korea', 'UK'], n_samples)
    })

    def label_fraud(row):
        if row['country'] in ['North Korea', 'Iran']:
            return 1
        if row['txn_type'] == 'crypto' and row['amount'] > 300:
            return 1
        if row['txn_type'] == 'international' and row['amount'] > 500:
            return 1
        return 0

    data['label'] = data.apply(label_fraud, axis=1)

    categorical = pd.get_dummies(data[['txn_type', 'country']], drop_first=True)
    features = data[['amount', 'time', 'location_score', 'device_score', 'user_age']].join(categorical)
    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    features['label'] = data['label'].values

    return features

# Step 2: Define Gymnasium-compatible custom environment
class AdvancedFraudEnv(gym.Env):
    def __init__(self, df):
        super(AdvancedFraudEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.action_space = spaces.Discrete(2)  # 0 = Valid, 1 = Fraud
        self.observation_space = spaces.Box(low=-5, high=5, shape=(df.shape[1] - 1,), dtype=np.float32)
        self.current_idx = 0

    def reset(self, seed=None, options=None):
        self.current_idx = 0
        return self.df.iloc[self.current_idx, :-1].values.astype(np.float32), {}

    def step(self, action):
        label = self.df.iloc[self.current_idx]['label']
        reward = 1 if action == label else -1
        self.current_idx += 1
        done = self.current_idx >= len(self.df)
        next_state = self.df.iloc[self.current_idx, :-1].values.astype(np.float32) if not done else np.zeros(self.df.shape[1] - 1, dtype=np.float32)
        return next_state, reward, done, False, {}

# Step 3: Load data and train or load PPO model
features_df = generate_data()

if not os.path.exists("ppo_fraud_advanced.zip"):
    env = AdvancedFraudEnv(features_df)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10000)
    model.save("ppo_fraud_advanced")
else:
    model = PPO.load("ppo_fraud_advanced")

# Step 4: Streamlit UI
st.set_page_config(page_title="RL FinTech Fraud Detector", layout="centered")
st.title("💸 RL-Powered FinTech Transaction Validator")

st.markdown("""
### Simulate a Transaction:
Fill in the transaction details below. The RL agent will predict if it should be accepted or flagged as fraud.
""")

amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
time_val = st.slider("Time of Day (24h Format)", 0, 23, 12)
location_score = st.slider("Location Trust Score", 0.0, 1.0, 0.5)
device_score = st.slider("Device Trust Score", 0.0, 1.0, 0.5)
user_age = st.slider("User Age", 18, 70, 30)
txn_type = st.selectbox("Transaction Type", ["crypto", "merchant", "international"])
country = st.selectbox("Transaction Country", ["USA", "India", "Germany", "Iran", "North Korea", "UK"])

if st.button("Submit Transaction"):
    input_df = pd.DataFrame([{
        'amount': amount,
        'time': time_val,
        'location_score': location_score,
        'device_score': device_score,
        'user_age': user_age,
        'txn_type': txn_type,
        'country': country
    }])

    cat_cols = pd.get_dummies(input_df[['txn_type', 'country']])
    all_columns = pd.get_dummies(features_df.drop(columns=['label'])).columns
    missing_cols = set(all_columns) - set(cat_cols.columns) - set(input_df.columns)
    for col in missing_cols:
        cat_cols[col] = 0

    numeric_cols = input_df[['amount', 'time', 'location_score', 'device_score', 'user_age']]
    full_input = pd.concat([numeric_cols, cat_cols], axis=1)[all_columns]
    full_input = pd.DataFrame(StandardScaler().fit_transform(full_input), columns=full_input.columns)

    action, _ = model.predict(full_input)

    with st.spinner("Analyzing transaction..."):
        time.sleep(1.5)

    if action[0] == 1:
        st.error("🚨 Transaction Flagged as FRAUD!")
    else:
        st.success("✅ Transaction Accepted as VALID.")

    st.markdown(f"**RL Agent Prediction:** {'Fraud' if action[0] == 1 else 'Valid'}")

    # ✅ Save transaction to CSV for Jupyter view
    log_data = input_df.copy()
    log_data['Prediction'] = 'Fraud' if action[0] == 1 else 'Valid'
    log_data['Timestamp'] = pd.Timestamp.now()

    # Save to CSV file
    if os.path.exists("submitted_transactions.csv"):
        log_data.to_csv("submitted_transactions.csv", mode='a', header=False, index=False)
    else:
        log_data.to_csv("submitted_transactions.csv", index=False)

st.markdown("---")
st.markdown("Made with ❤️ using PPO, Gymnasium, and Streamlit")
