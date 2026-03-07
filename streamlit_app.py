import os
import numpy as np
import librosa
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from tensorflow.keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model/model.keras"
CLASS_PATH = "model/categories.pkl"

MAX_LEN = 100

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_ai_model()

with open(CLASS_PATH, "rb") as f:
    class_names = pickle.load(f)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(audio, sr):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=128
    )

    mel = librosa.power_to_db(mel)

    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0,0),(0,MAX_LEN-mfcc.shape[1])))
    else:
        mfcc = mfcc[:,:MAX_LEN]

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(mel, ((0,0),(0,MAX_LEN-mel.shape[1])))
    else:
        mel = mel[:,:MAX_LEN]

    return mfcc, mel

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🫁 Respiratory Disease Detection from Audio")

st.write("Upload a respiratory sound recording to classify disease.")

uploaded_file = st.file_uploader(
    "Upload audio",
    type=["wav","mp3","ogg","flac","m4a"]
)

if uploaded_file:

    st.audio(uploaded_file)

    audio, sr = librosa.load(uploaded_file, sr=None)

    mfcc, mel = extract_features(audio, sr)

    mfcc_input = np.expand_dims(mfcc, axis=(0,-1))
    mel_input = np.expand_dims(mel, axis=(0,-1))

    if st.button("Run Prediction"):

        probs = model.predict([mfcc_input, mel_input])[0]

        pred_idx = np.argmax(probs)
        pred_label = class_names[pred_idx]

        st.success(f"Prediction: {pred_label}")

        # -----------------------------
        # Probability Chart
        # -----------------------------
        st.subheader("Prediction Probabilities")

        prob_dict = {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }

        st.bar_chart(prob_dict)

        # -----------------------------
        # MFCC Visualization
        # -----------------------------
        st.subheader("MFCC Features")

        fig1, ax1 = plt.subplots()

        ax1.imshow(mfcc, aspect="auto", cmap="viridis")
        ax1.set_title("MFCC")

        st.pyplot(fig1)

        # -----------------------------
        # MEL Spectrogram
        # -----------------------------
        st.subheader("Mel Spectrogram")

        fig2, ax2 = plt.subplots()

        ax2.imshow(mel, aspect="auto", cmap="viridis")
        ax2.set_title("Mel Spectrogram")

        st.pyplot(fig2)