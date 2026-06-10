import base64
import json
import os
import random
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image
from sklearn.datasets import load_breast_cancer

from agents.agent import CancerAgent

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
TEMPLATES_DIR = os.path.join(PROJECT_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATES_DIR)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

cnn_model = None
multimodal_model = None

agent = CancerAgent()

with open(os.path.join(PROJECT_DIR, "model_results.json"), "r", encoding="utf-8") as f:
    MODEL_RESULTS = json.load(f)

with open(os.path.join(PROJECT_DIR, "cnn_metrics.json"), "r", encoding="utf-8") as f:
    CNN_METRICS = json.load(f)

_mm_path = os.path.join(PROJECT_DIR, "multimodal_metrics.json")
MULTIMODAL_METRICS = json.load(open(_mm_path)) if os.path.exists(_mm_path) else {}

SCALER = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
FEATURE_NAMES = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
MODELS = {
    "Random Forest": joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl")),
    "Gradient Boosting": joblib.load(os.path.join(MODELS_DIR, "gradient_boosting.pkl")),
    "SVM": joblib.load(os.path.join(MODELS_DIR, "svm.pkl")),
    "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl")),
    "KNN": joblib.load(os.path.join(MODELS_DIR, "knn.pkl")),
}

DATA = load_breast_cancer(as_frame=True)
DF = DATA.frame
FEATURE_MEANS = DF[FEATURE_NAMES].mean()
FEATURE_STDS = DF[FEATURE_NAMES].std()
FEATURE_EXAMPLES = {name: float(round(FEATURE_MEANS[name], 3)) for name in FEATURE_NAMES}
TARGET_LABELS = {0: "Malignant", 1: "Benign"}


def lazy_load_cnn():
    global cnn_model
    if cnn_model is None:
        import tensorflow as tf

        model_path = os.path.join(MODELS_DIR, "cnn_model.keras")
        cnn_model = tf.keras.models.load_model(model_path)
    return cnn_model


def lazy_load_multimodal():
    global multimodal_model
    if multimodal_model is None:
        import tensorflow as tf
        model_path = os.path.join(MODELS_DIR, "multimodal_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Multimodal model not found. Run multimodal_train.py first."
            )
        multimodal_model = tf.keras.models.load_model(model_path)
    return multimodal_model


def preprocess_for_multimodal(image_bytes):
    """Resize to 224x224 RGB for EfficientNetB0."""
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.reshape(1, 224, 224, 3)
    thumb = image.resize((128, 128), Image.LANCZOS)
    buf = BytesIO()
    thumb.save(buf, format="PNG")
    preview = base64.b64encode(buf.getvalue()).decode("utf-8")
    return array, preview


def dataset_stats():
    target_counts = DF[DATA.target_names].copy() if False else None
    counts = DF["target"].value_counts().to_dict()
    return {
        "samples": int(DF.shape[0]),
        "features": len(FEATURE_NAMES),
        "classes": [name for name in DATA.target_names],
        "distribution": {
            TARGET_LABELS[key]: int(value)
            for key, value in counts.items()
        },
        "feature_groups": {
            "mean": [name for name in FEATURE_NAMES if "mean" in name],
            "se": [name for name in FEATURE_NAMES if "se" in name],
            "worst": [name for name in FEATURE_NAMES if "worst" in name],
        },
    }


def normalize_features(feature_vector: np.ndarray) -> np.ndarray:
    values = np.array(feature_vector, dtype=float).reshape(1, -1)
    return SCALER.transform(values)


def risk_level(probability: float) -> str:
    if probability >= 0.85:
        return "High"
    if probability >= 0.65:
        return "Moderate"
    return "Low"


def flagged_features(feature_vector: dict) -> list:
    flags = []
    for name, value in feature_vector.items():
        if name not in FEATURE_MEANS:
            continue
        diff = abs(value - FEATURE_MEANS[name]) / (FEATURE_STDS[name] + 1e-9)
        if diff >= 1.3:
            direction = "higher than" if value > FEATURE_MEANS[name] else "lower than"
            flags.append({
                "feature": name,
                "value": float(value),
                "comparison": f"{direction} average",
                "deviation": float(round(diff, 2)),
            })
    return flags[:6]


def build_prediction_payload(model_name, features):
    scaled = normalize_features(features)
    model = MODELS.get(model_name)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")
    proba = model.predict_proba(scaled)[0]
    prediction = int(model.predict(scaled)[0])
    label = TARGET_LABELS[prediction]
    probability = float(round(proba[1] if prediction == 1 else 1 - proba[1], 4))
    risk = risk_level(1 - proba[1] if prediction == 0 else proba[1])
    feature_map = {name: float(features[idx]) for idx, name in enumerate(FEATURE_NAMES)}
    report = agent.generate_report(label, probability, risk, feature_map)
    return {
        "model": model_name,
        "label": label,
        "prediction": prediction,
        "probabilities": {
            "Benign": float(round(proba[1], 4)),
            "Malignant": float(round(proba[0], 4)),
        },
        "confidence": float(round(max(proba) * 100, 2)),
        "risk_level": risk,
        "flagged_features": flagged_features(feature_map),
        "analysis": report,
    }


def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("L")
    image = image.resize((64, 64), Image.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = array.reshape(1, 64, 64, 1)
    thumb = image.resize((128, 128), Image.LANCZOS)
    buffered = BytesIO()
    thumb.save(buffered, format="PNG")
    preview = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return array, preview


def predict_image_content(image_bytes):
    model = lazy_load_cnn()
    array, preview = preprocess_image(image_bytes)
    proba = model.predict(array, verbose=0)[0][0]
    label = "Benign" if proba >= 0.5 else "Malignant"
    confidence = float(round(max(proba, 1 - proba) * 100, 2))
    risk = risk_level(proba)
    recommendations = agent.generate_report(label, float(round(proba, 4)), risk, {})
    return {
        "label": label,
        "probability": float(round(proba, 4)),
        "confidence": confidence,
        "risk_level": risk,
        "probabilities": {
            "Benign": float(round(proba, 4)),
            "Malignant": float(round(1 - proba, 4)),
        },
        "analysis": recommendations,
        "thumbnail": f"data:image/png;base64,{preview}",
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Breast Cancer AI API is running."})


@app.route("/api/feature_names", methods=["GET"])
def feature_names():
    return jsonify({
        "feature_names": list(FEATURE_NAMES),
        "feature_examples": FEATURE_EXAMPLES,
    })


@app.route("/api/dataset_stats", methods=["GET"])
def get_dataset_stats():
    return jsonify(dataset_stats())


@app.route("/api/model_results", methods=["GET"])
def get_model_results():
    return jsonify(MODEL_RESULTS)


@app.route("/api/compare_models", methods=["GET"])
def compare_models():
    sorted_results = sorted(MODEL_RESULTS["models"], key=lambda item: item["accuracy"], reverse=True)
    return jsonify({"models": sorted_results})


@app.route("/api/cnn_metrics", methods=["GET"])
def cnn_metrics_route():
    return jsonify(CNN_METRICS)


@app.route("/api/multimodal_metrics", methods=["GET"])
def multimodal_metrics_route():
    return jsonify(MULTIMODAL_METRICS)


@app.route("/api/predict_multimodal", methods=["POST"])
def predict_multimodal():
    """
    Multimodal prediction using EfficientNetB0 fusion model.
    Accepts form-data with up to 3 image files:
        ultrasound, histopathology, chest_xray
    At least one must be provided; missing ones are zero-padded.
    """
    ZERO = np.zeros((1, 224, 224, 3), dtype=np.float32)
    inputs, previews = [], {}

    for modality in ["ultrasound", "histopathology", "chest_xray"]:
        f = request.files.get(modality)
        if f and f.filename:
            arr, prev = preprocess_for_multimodal(f.read())
            inputs.append(arr)
            previews[modality] = f"data:image/png;base64,{prev}"
        else:
            inputs.append(ZERO)
            previews[modality] = None

    if all(np.array_equal(x, ZERO) for x in inputs):
        return jsonify({"error": "Provide at least one image (ultrasound, histopathology, or chest_xray)."}), 400

    try:
        model = lazy_load_multimodal()
        proba = float(model.predict(inputs, verbose=0)[0][0])
        label = "Benign" if proba >= 0.5 else "Malignant"
        risk  = risk_level(proba)
        analysis = agent.generate_report(label, round(proba, 4), risk, {})
        return jsonify({
            "label":               label,
            "probability":         round(proba, 4),
            "confidence":          round(max(proba, 1 - proba) * 100, 2),
            "risk_level":          risk,
            "probabilities":       {"Benign": round(proba, 4), "Malignant": round(1 - proba, 4)},
            "modalities_provided": [m for m in ["ultrasound", "histopathology", "chest_xray"] if previews[m]],
            "thumbnails":          previews,
            "analysis":            analysis,
            "model":               "Multimodal EfficientNetB0 (US + Histo + CXR)",
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/api/dataset_samples", methods=["GET"])
def dataset_samples():
    count = request.args.get('count', 5, type=int)
    count = min(count, 20)
    indices = np.random.choice(DF.shape[0], size=count, replace=False)
    samples = []
    for idx in indices:
        sample = DF.iloc[idx]
        features = sample[FEATURE_NAMES].tolist()
        samples.append({
            "index": int(idx),
            "features": features,
            "target": TARGET_LABELS[int(sample["target"])],
        })
    return jsonify({"samples": samples})


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    model_name = payload.get("model") or "Random Forest"
    features = payload.get("features")
    if not features or len(features) != len(FEATURE_NAMES):
        return jsonify({"error": "features must be a list of 30 numerical values"}), 400
    try:
        result = build_prediction_payload(model_name, features)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Uploaded file has no filename."}), 400
    try:
        content = file.read()
        result = predict_image_content(content)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Failed to process image: {exc}"}), 500


@app.route("/api/predict_sample", methods=["GET"])
def predict_sample():
    idx = random.randint(0, DF.shape[0] - 1)
    sample = DF.iloc[idx]
    features = sample[FEATURE_NAMES].tolist()
    sample_result = {
        "sample_index": int(idx),
        "features": {name: float(sample[name]) for name in FEATURE_NAMES},
        "target": TARGET_LABELS[int(sample["target"])],
        "predictions": [build_prediction_payload(name, features) for name in MODELS.keys()],
    }
    sample_result["agent_report"] = agent.generate_report_for_sample(sample_result)
    return jsonify(sample_result)


@app.route("/api/agent/chat", methods=["POST"])
def agent_chat():
    payload = request.get_json(force=True)
    question = payload.get("question", "").strip()
    context = payload.get("context", {})
    if not question:
        return jsonify({"error": "Question is required."}), 400
    answer = agent.chat(question, context)
    return jsonify({"question": question, "answer": answer})


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Uploaded file is too large. Max size is 16 MB."}), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
