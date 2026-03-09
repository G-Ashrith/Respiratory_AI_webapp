import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pickle
import io
import keras
st.set_page_config(page_title="Respiratory Sound Classifier", layout="wide")

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

MODEL_FILE = "model/model.keras"
CATEGORIES_FILE = "model/categories.pkl"

MAX_LEN = 100

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model_files():

    model = keras.models.load_model(
    MODEL_FILE,
    compile=False
)

    with open(CATEGORIES_FILE, "rb") as f:
        raw_categories = pickle.load(f)

    if isinstance(raw_categories, (list, tuple, np.ndarray)):
        class_names = list(sorted(set(raw_categories)))

    elif isinstance(raw_categories, dict) and "classes_" in raw_categories:
        class_names = list(raw_categories["classes_"])

    else:
        class_names = list(sorted(set(raw_categories)))

    return model, class_names


model, class_names = load_model_files()

# ------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------

def extract_mfcc(file_path, n_mfcc=40):

    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
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
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc


def extract_mel(file_path, n_mels=128):

    y, sr = librosa.load(file_path, sr=None)

    mel = librosa.feature.melspectrogram(
        y=y,
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
        mel = mel[:, :MAX_LEN]

    return mel


# ------------------------------------------------
# GRADCAM SETUP
# ------------------------------------------------

conv_layers = [
    l.name for l in model.layers
    if isinstance(l, tf.keras.layers.Conv2D)
]

LAST_CONV_NAME = conv_layers[-2]


# ------------------------------------------------
# GUIDED GRADCAM
# ------------------------------------------------

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


# ------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------

st.title("Respiratory Sound Classification with Explainability")

uploaded_file = st.file_uploader(
    "Upload Respiratory Audio",
    type=["wav","mp3","ogg","flac","m4a"]
)

if uploaded_file:

    st.audio(uploaded_file)

    # Save temporary file
    with open("temp_audio.wav","wb") as f:
        f.write(uploaded_file.read())

    # ------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------

    mfcc = extract_mfcc("temp_audio.wav").astype(np.float32)
    mel = extract_mel("temp_audio.wav").astype(np.float32)

    mfcc_input = np.expand_dims(mfcc, axis=(0,-1))
    mel_input = np.expand_dims(mel, axis=(0,-1))

    # ------------------------------------------------
    # PREDICTION
    # ------------------------------------------------

    probs = model.predict([mfcc_input, mel_input])[0]

    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]

    st.success(f"Prediction: **{pred_label}**")

    # Probability chart
    st.subheader("Prediction Probabilities")

    fig, ax = plt.subplots()

    ax.bar(class_names, probs)

    ax.set_ylabel("Probability")
    ax.set_ylim(0,1)

    st.pyplot(fig)

    # ------------------------------------------------
    # GRADCAM
    # ------------------------------------------------

    heatmap = guided_gradcam(
        model,
        mfcc_input,
        mel_input,
        pred_idx
    )

    heatmap = tf.nn.avg_pool2d(
        heatmap[np.newaxis,...,np.newaxis],
        ksize=3,
        strides=1,
        padding="SAME"
    )[0,...,0]

    heat_mfcc = tf.image.resize(
        heatmap[...,np.newaxis],
        (mfcc.shape[0], mfcc.shape[1]),
        method="bicubic"
    ).numpy().squeeze()

    heat_mel = tf.image.resize(
        heatmap[...,np.newaxis],
        (mel.shape[0], mel.shape[1]),
        method="bicubic"
    ).numpy().squeeze()

    heat_mfcc = (heat_mfcc-heat_mfcc.min())/(heat_mfcc.max()-heat_mfcc.min()+1e-8)
    heat_mel = (heat_mel-heat_mel.min())/(heat_mel.max()-heat_mel.min()+1e-8)

    # ------------------------------------------------
    # EXPLAINABILITY PANEL
    # ------------------------------------------------

    mfcc_norm = (mfcc-np.mean(mfcc))/(np.std(mfcc)+1e-8)
    mel_norm = (mel-mel.min())/(mel.max()-mel.min()+1e-8)

    st.subheader("Explainability (Guided Grad-CAM)")

    fig, axes = plt.subplots(2,2, figsize=(12,6))

    axes[0,0].imshow(mfcc_norm,cmap="gray",aspect="auto")
    axes[0,0].set_title("MFCC")

    axes[0,1].imshow(mfcc_norm,cmap="gray",aspect="auto")
    axes[0,1].imshow(heat_mfcc,cmap="jet",alpha=0.75,aspect="auto")
    axes[0,1].set_title("MFCC + GradCAM")

    axes[1,0].imshow(mel_norm,cmap="gray",aspect="auto")
    axes[1,0].set_title("Mel Spectrogram")

    axes[1,1].imshow(mel_norm,cmap="gray",aspect="auto")
    axes[1,1].imshow(heat_mel,cmap="jet",alpha=0.75,aspect="auto")
    axes[1,1].set_title("Mel + GradCAM")

    plt.tight_layout()

    st.pyplot(fig)