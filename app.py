import os
import io
import uuid
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import numpy as np
import librosa

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

import pickle


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_FILE = os.path.join(MODEL_DIR, "model.h5")
CATEGORIES_FILE = os.path.join(MODEL_DIR, "categories.pkl")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")

ALLOWED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

MAX_LEN = 100

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ------------------------------------------------
# FLASK
# ------------------------------------------------

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

print("Loading model...")
model = load_model(MODEL_FILE, compile=False)
print("Model loaded.")


with open(CATEGORIES_FILE, "rb") as f:
    raw_categories = pickle.load(f)

if isinstance(raw_categories, (list, tuple, np.ndarray)):
    class_names = list(sorted(set(raw_categories)))

elif isinstance(raw_categories, dict) and "classes_" in raw_categories:
    class_names = list(raw_categories["classes_"])

else:
    class_names = list(sorted(set(raw_categories)))

print("Classes:", class_names)


# ------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------

def extract_mfcc(file_path, n_mfcc=40, max_len=MAX_LEN):

    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc
    )

    if mfcc.shape[1] < max_len:

        mfcc = np.pad(
            mfcc,
            ((0,0),(0, max_len - mfcc.shape[1])),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :max_len]

    return mfcc



def extract_mel(file_path, n_mels=128, max_len=MAX_LEN):

    y, sr = librosa.load(file_path, sr=None)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    if mel.shape[1] < max_len:

        mel = np.pad(
            mel,
            ((0,0),(0, max_len - mel.shape[1])),
            mode="constant"
        )

    else:

        mel = mel[:, :max_len]

    return mel



def allowed_file(filename):

    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT



# ------------------------------------------------
# GRADCAM UTILITIES
# ------------------------------------------------

# ------------------------------------------------
# Choose convolution layer for Grad-CAM
# ------------------------------------------------

conv_layers = [
    l.name for l in model.layers
    if isinstance(l, tf.keras.layers.Conv2D)
]

print("All Conv layers:", conv_layers)

# Use second-last conv layer for better spatial resolution
LAST_CONV_NAME = conv_layers[-2]

print("Grad-CAM layer selected:", LAST_CONV_NAME)



# ------------------------------------------------
# GUIDED GRAD-CAM
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
# EXPLAINABILITY PANEL
# ------------------------------------------------

def generate_explainability_panel(mfcc_img, mel_img, heat_mfcc, heat_mel):

    mfcc_norm = (mfcc_img - np.mean(mfcc_img)) / (np.std(mfcc_img) + 1e-8)

    mel_norm = (mel_img - mel_img.min()) / (
        mel_img.max() - mel_img.min() + 1e-8
    )


    fig, axes = plt.subplots(2,2, figsize=(12,6))


    # MFCC

    axes[0,0].imshow(
        mfcc_norm,
        cmap="gray",
        aspect="auto"
    )

    axes[0,0].set_title("MFCC")

    axes[0,0].set_xlabel("Time")

    axes[0,0].set_ylabel("MFCC")



    # MFCC + CAM

    axes[0,1].imshow(
        mfcc_norm,
        cmap="gray",
        aspect="auto"
    )

    axes[0,1].imshow(
        heat_mfcc,
        cmap="jet",
        alpha=0.75,
        interpolation="bilinear",
        vmin=0.2,
        vmax=1,
        aspect="auto"
    )

    axes[0,1].set_title("MFCC + Guided Grad-CAM")

    axes[0,1].set_xlabel("Time")

    axes[0,1].set_ylabel("MFCC")



    # MEL

    axes[1,0].imshow(
        mel_norm,
        cmap="gray",
        aspect="auto"
    )

    axes[1,0].set_title("Mel Spectrogram")

    axes[1,0].set_xlabel("Time")

    axes[1,0].set_ylabel("Mel Frequency")



    # MEL + CAM

    axes[1,1].imshow(
        mel_norm,
        cmap="gray",
        aspect="auto"
    )

    axes[1,1].imshow(
        heat_mel,
        cmap="jet",
        alpha=0.75,
        interpolation="bilinear",
        vmin=0.2,
        vmax=1,
        aspect="auto"
    )

    axes[1,1].set_title("Mel + Guided Grad-CAM")

    axes[1,1].set_xlabel("Time")

    axes[1,1].set_ylabel("Mel Frequency")


    plt.tight_layout()


    buf = io.BytesIO()

    plt.savefig(buf, format="png")

    plt.close()


    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode("utf-8")



# ------------------------------------------------
# ROUTES
# ------------------------------------------------

@app.route("/")
def index():

    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:

        return jsonify({"error":"No file uploaded"}), 400


    f = request.files["file"]


    if f.filename == "":

        return jsonify({"error":"No file selected"}), 400


    if not allowed_file(f.filename):

        return jsonify({"error":"Unsupported file"}), 400


    fname = secure_filename(f.filename)

    uid = str(uuid.uuid4())

    path = os.path.join(
        UPLOAD_FOLDER,
        uid + "_" + fname
    )

    f.save(path)



    # ------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------

    mfcc = extract_mfcc(path).astype(np.float32)

    mel = extract_mel(path).astype(np.float32)


    mfcc_input = np.expand_dims(mfcc, axis=(0,-1))

    mel_input = np.expand_dims(mel, axis=(0,-1))



    # ------------------------------------------------
    # PREDICTION
    # ------------------------------------------------

    try:

        probs = model.predict([mfcc_input, mel_input])[0]

    except:

        inp_names = [x.name.split(":")[0] for x in model.inputs]

        inp_dict = {
            inp_names[0]: mfcc_input,
            inp_names[1]: mel_input
        }

        probs = model.predict(inp_dict)[0]


    pred_idx = int(np.argmax(probs))

    pred_label = class_names[pred_idx]



    # ------------------------------------------------
    # GUIDED GRADCAM
    # ------------------------------------------------

    heatmap = guided_gradcam(
        model,
        mfcc_input,
        mel_input,
        pred_idx
    )


    # smooth heatmap

    heatmap = tf.nn.avg_pool2d(

        heatmap[np.newaxis,...,np.newaxis],

        ksize=3,

        strides=1,

        padding="SAME"

    )[0,...,0]


    mfcc_img = mfcc

    mel_img = mel


    heat_mfcc = tf.image.resize(

        heatmap[...,np.newaxis],

        (mfcc_img.shape[0], mfcc_img.shape[1]),

        method="bicubic"

    ).numpy().squeeze()



    heat_mel = tf.image.resize(

        heatmap[...,np.newaxis],

        (mel_img.shape[0], mel_img.shape[1]),

        method="bicubic"

    ).numpy().squeeze()



    heat_mfcc = (

        heat_mfcc - heat_mfcc.min()

    ) / (

        heat_mfcc.max() - heat_mfcc.min() + 1e-8

    )


    heat_mel = (

        heat_mel - heat_mel.min()

    ) / (

        heat_mel.max() - heat_mel.min() + 1e-8

    )



    # ------------------------------------------------
    # CREATE PANEL
    # ------------------------------------------------

    panel_b64 = generate_explainability_panel(

        mfcc_img,
        mel_img,
        heat_mfcc,
        heat_mel

    )



    response = {

        "pred_label": pred_label,

        "pred_index": pred_idx,

        "probs": {
            name: float(p)
            for name,p in zip(class_names, probs)
        },

        "explainability_panel":
        "data:image/png;base64," + panel_b64

    }


    return jsonify(response)



# ------------------------------------------------

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)