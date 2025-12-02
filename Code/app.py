#!/usr/bin/env python3
"""
Streamlit App for Figurative Language Detection on Tweets

Models supported:
  - TFIDF + Logistic Regression   (models/baseline)
  - BiLSTM                        (models/bilstm)
  - DistilBERT                    (models/distilbert)  — if transformers/torch work

Input:
  - User enters a tweet in a text box

Output:
  - Predicted label: figurative / irony / regular / sarcasm
  - Class probabilities table
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Try importing HF/torch for DistilBERT, but handle if not available
try:
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# -----------------------------
# CONFIG
# -----------------------------
BASELINE_DIR = "models/baseline"
BILSTM_DIR = "models/bilstm"
DISTILBERT_DIR = "models/distilbert"

MAX_LEN_LSTM = 50   # must match your BiLSTM training script
TEXT_PLACEHOLDER = "Type a tweet here, e.g. 'Yeah sure, this is exactly what I needed today...'"


# -----------------------------
# CACHE HELPERS
# -----------------------------
@st.cache_resource
def load_baseline_model():
    tfidf = joblib.load(os.path.join(BASELINE_DIR, "tfidf_vectorizer.joblib"))
    clf = joblib.load(os.path.join(BASELINE_DIR, "logreg_baseline.joblib"))
    le = joblib.load(os.path.join(BASELINE_DIR, "label_encoder.joblib"))
    return tfidf, clf, le


@st.cache_resource
def load_bilstm_model():
    model = load_model(os.path.join(BILSTM_DIR, "final_lstm_model.h5"))
    with open(os.path.join(BILSTM_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(BILSTM_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le


@st.cache_resource
def load_distilbert_model():
    if not HF_AVAILABLE:
        raise RuntimeError("Transformers / torch not available in this environment.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DistilBertForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_DIR)
    model.to(device)
    model.eval()
    label2id = model.config.label2id
    id2label = model.config.id2label
    return model, tokenizer, label2id, id2label, device


# -----------------------------
# PREDICTION HELPERS
# -----------------------------
def predict_baseline(text: str):
    tfidf, clf, le = load_baseline_model()
    X = tfidf.transform([text])
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = le.classes_[pred_idx]
    return pred_label, probs, list(le.classes_)


def predict_bilstm(text: str):
    model, tokenizer, le = load_bilstm_model()
    seq = pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=MAX_LEN_LSTM,
        padding="post",
        truncating="post",
    )
    probs = model.predict(seq, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]
    return pred_label, probs, list(le.classes_)


def predict_distilbert(text: str):
    model, tokenizer, label2id, id2label, device = load_distilbert_model()
    enc = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = id2label[pred_idx]
    class_names = [id2label[i] for i in range(len(id2label))]
    return pred_label, probs, class_names


# -----------------------------
# STREAMLIT UI
# -----------------------------
def main():
    st.set_page_config(page_title="Figurative Language Detection", layout="centered")

    st.title("🗣️ Figurative Language Detection in Tweets")
    st.write(
        "Classify a tweet as **figurative**, **irony**, **regular**, or **sarcasm** "
        "using the models you trained (TFIDF+LogReg, BiLSTM, DistilBERT)."
    )

    # Sidebar
    st.sidebar.header("Model Settings")
    model_options = ["TFIDF + Logistic Regression", "BiLSTM"]
    if HF_AVAILABLE:
        model_options.append("DistilBERT")
    model_choice = st.sidebar.radio("Choose model:", model_options)

    if not HF_AVAILABLE:
        st.sidebar.warning("DistilBERT disabled (transformers/torch not available in this env).")

    # Main input
    text = st.text_area(
        "Tweet text",
        value="",
        height=120,
        placeholder=TEXT_PLACEHOLDER,
    )

    if st.button("Predict"):
        if not text.strip():
            st.warning("Please enter a tweet before predicting.")
            return

        with st.spinner("Running model..."):
            try:
                if model_choice == "TFIDF + Logistic Regression":
                    pred_label, probs, class_names = predict_baseline(text)

                elif model_choice == "BiLSTM":
                    pred_label, probs, class_names = predict_bilstm(text)

                else:  # DistilBERT
                    pred_label, probs, class_names = predict_distilbert(text)

            except Exception as e:
                st.error(f"Error while running {model_choice}: {e}")
                return

        # Show result
        st.subheader("Prediction")
        st.markdown(f"**Predicted label:** `{pred_label}`")

        # Probabilities table
        st.subheader("Class Probabilities")
        df_probs = pd.DataFrame(
            {"Class": class_names, "Probability": probs}
        ).sort_values("Probability", ascending=False)
        st.table(df_probs.style.format({"Probability": "{:.4f}"}))

        # Small chart (Streamlit's built-in bar chart; you can comment this out if you hate bars 😂)
        st.bar_chart(df_probs.set_index("Class"))

    st.markdown("---")
    st.caption(
        "Backend models: TFIDF+LogReg, BiLSTM, DistilBERT, trained on figurative/irony/sarcasm/regular tweet corpus."
    )


if __name__ == "__main__":
    main()
