import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection")
st.write("Upload news text or paste content to predict whether it is Fake or Real.")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except:
    model = None

news_text = st.text_area("Enter News Content", height=250)

if st.button("Predict"):
    if not news_text.strip():
        st.warning("Please enter some text.")
    elif model is None:
        st.error("model.pkl not found. Run model_training.py first.")
    else:
        prediction = model.predict([news_text])[0]
        if prediction == 1:
            st.success("Prediction: Real News ✅")
        else:
            st.error("Prediction: Fake News ❌")

st.markdown("---")
st.caption("Dataset: Kaggle Fake News Detection")
