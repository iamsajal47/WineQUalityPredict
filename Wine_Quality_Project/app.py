import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

@st.cache_data
def load_data(csv_path='winequality-red.csv'):
    # Check if file exists
    if not os.path.exists(csv_path):
        st.error(f"❌ File '{csv_path}' not found. Please upload it below.")
        return None
    df = pd.read_csv(csv_path)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    return model, scaler, acc

# ---------------- Streamlit App ----------------
st.title("🍷 Wine Quality Prediction")

# File upload option
uploaded_file = st.file_uploader("Upload your wine quality CSV file", type=["csv"])

# If uploaded, save it temporarily
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()  # tries to load default file

if df is not None:
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    model, scaler, acc = train_model(df)
    st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

    st.write("### Make a Prediction")
    feature_inputs = []
    for col in df.columns[:-1]:  # skip 'quality'
        value = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        feature_inputs.append(value)

    if st.button("Predict Quality"):
        scaled_input = scaler.transform([feature_inputs])
        prediction = model.predict(scaled_input)
        st.write(f"### 🍾 Predicted Wine Quality: **{prediction[0]}**")

else:
    st.warning("Please upload 'winequality-red.csv' to continue.")
