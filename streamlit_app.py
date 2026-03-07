import os
import io
import base64
import numpy as np
import librosa
import pickle
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# -----------------------------
# CONFIG
# -----------------------------
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "model.h5")
CATEGORIES_FILE = os.path.join(MODEL_DIR, "categories.pkl")

MAX_LEN = 100

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_ai_model():
    model = load_model(MODEL_FILE)
    return model

model = load_ai_model()

with open(CATEGORIES_FILE, "rb") as f:
    raw_categories = pickle.load(f)

if isinstance(raw_categories, (list, tuple, np.ndarray)):
    class_names = list(sorted(set(raw_categories)))
else:
    class_names = list(raw_categories)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_mfcc(audio, sr, n_mfcc=40):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc
    )

    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(
            mfcc,
            ((0,0),(0,MAX_LEN-mfcc.shape[1])),
            mode="constant"
        )
    else:
        mfcc = mfcc[:,:MAX_LEN]

    return mfcc


def extract_mel(audio, sr, n_mels=128):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < MAX_LEN:
        mel = np.pad(
            mel,
            ((0,0),(0,MAX_LEN-mel.shape[1])),
            mode="constant"
        )
    else:
        mel = mel[:,:MAX_LEN]

    return mel

# -----------------------------
# GRADCAM
# -----------------------------
conv_layers = [
    l.name for l in model.layers
    if isinstance(l, tf.keras.layers.Conv2D)
]

LAST_CONV_NAME = conv_layers[-2]


def guided_gradcam(model, mfcc_input, mel_input, class_index):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(LAST_CONV_NAME).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            [mfcc_input, mel_input],
            training=False
        )

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    guided_grads = (
        tf.cast(conv_outputs > 0, "float32")
        * tf.cast(grads > 0, "float32")
        * grads
    )

    weights = tf.reduce_mean(guided_grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    cam = tf.reduce_sum(weights * conv_outputs, axis=-1)

    cam = tf.maximum(cam, 0)
    cam /= tf.reduce_max(cam) + 1e-8

    return cam.numpy()

# -----------------------------
# EXPLAINABILITY PANEL
# -----------------------------
def generate_panel(mfcc_img, mel_img, heat_mfcc, heat_mel):

    fig, axes = plt.subplots(2,2, figsize=(10,6))

    axes[0,0].imshow(mfcc_img, aspect="auto", cmap="gray")
    axes[0,0].set_title("MFCC")

    axes[0,1].imshow(mfcc_img, cmap="gray", aspect="auto")
    axes[0,1].imshow(heat_mfcc, cmap="jet", alpha=0.7, aspect="auto")
    axes[0,1].set_title("MFCC + GradCAM")

    axes[1,0].imshow(mel_img, cmap="gray", aspect="auto")
    axes[1,0].set_title("Mel Spectrogram")

    axes[1,1].imshow(mel_img, cmap="gray", aspect="auto")
    axes[1,1].imshow(heat_mel, cmap="jet", alpha=0.7, aspect="auto")
    axes[1,1].set_title("Mel + GradCAM")

    plt.tight_layout()

    return fig

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Respiratory Disease Detection from Audio")

uploaded_file = st.file_uploader(
    "Upload Respiratory Audio",
    type=["wav","mp3","ogg","flac","m4a"]
)

if uploaded_file:

    st.audio(uploaded_file)

    audio, sr = librosa.load(uploaded_file, sr=None)

    mfcc = extract_mfcc(audio, sr)
    mel = extract_mel(audio, sr)

    mfcc_input = np.expand_dims(mfcc, axis=(0,-1))
    mel_input = np.expand_dims(mel, axis=(0,-1))

    if st.button("Predict"):

        probs = model.predict([mfcc_input, mel_input])[0]

        pred_idx = np.argmax(probs)
        pred_label = class_names[pred_idx]

        st.success(f"Prediction: {pred_label}")

        prob_dict = {
            name: float(p)
            for name,p in zip(class_names, probs)
        }

        st.subheader("Class Probabilities")
        st.json(prob_dict)

        # GradCAM
        heatmap = guided_gradcam(
            model,
            mfcc_input,
            mel_input,
            pred_idx
        )

        heat_mfcc = tf.image.resize(
            heatmap[...,np.newaxis],
            (mfcc.shape[0], mfcc.shape[1])
        ).numpy().squeeze()

        heat_mel = tf.image.resize(
            heatmap[...,np.newaxis],
            (mel.shape[0], mel.shape[1])
        ).numpy().squeeze()

        fig = generate_panel(
            mfcc,
            mel,
            heat_mfcc,
            heat_mel
        )

        st.pyplot(fig)