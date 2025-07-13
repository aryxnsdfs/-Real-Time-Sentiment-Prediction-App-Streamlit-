import streamlit as st
import pandas as pd
from joblib import load
from datetime import datetime
import time
import os

# Load model and vectorizer
model = load(r'C:\Users\aryan\Downloads\sentiment_model.joblib')
vectorizer = load(r'C:\Users\aryan\Downloads\vectorizer.joblib')

# Streamlit UI
st.set_page_config('predictor of postive and negative')
st.title('predicitor')


placeholder=st.empty()

def predict_sentiment(df):
    if "review" not in df.columns:
        return None
    df = df.dropna(subset=["review"])
    X = vectorizer.transform(df["review"])
    df["prediction"] = model.predict(X)
    df["label"] = df["prediction"].map({0: "Negative", 1: "Positive"})
    return df

while True:
    try:
        df = pd.read_csv(r"C:\Users\aryan\Downloads\TestRevs.csv")
        df.columns = df.columns.str.strip().str.lower()

        result_df = predict_sentiment(df)

        if result_df is not None:
            with placeholder.container():
                st.subheader("🆕 Latest Predictions (Last 5)")
                st.dataframe(result_df[["review", "label"]].tail(5), use_container_width=True)
                st.info(f"✅ Last updated: {datetime.now().strftime('%H:%M:%S')}")
        else:
            placeholder.warning("⚠️ Column 'review' not found.")
    
    except FileNotFoundError:
        placeholder.warning('no file found')

    time.sleep(5)





