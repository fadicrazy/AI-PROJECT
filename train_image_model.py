import json
import os

import numpy as np
import joblib
from PIL import Image
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_synthetic_images(features):
    n_samples = features.shape[0]
    grid = np.linspace(-1.0, 1.0, 64)
    x, y = np.meshgrid(grid, grid)
    images = np.zeros((n_samples, 64, 64), dtype=np.float32)
    for idx, vector in enumerate(features):
        seed = np.mean(vector) * 10.0
        image = np.zeros((64, 64), dtype=np.float32)
        for feat_index, value in enumerate(vector):
            frequency = (feat_index % 6) + 1
            pattern = np.sin((x * frequency + seed) * np.pi) * np.cos((y * (frequency + 1) - seed) * np.pi)
            image += pattern * (value + 1.0)
        image = image - image.min()
        image = image / (image.max() + 1e-9)
        images[idx] = image
    return images


def make_model():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(
        [
            layers.Input(shape=(64, 64, 1)),
            layers.Conv2D(24, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(48, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(96, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.32),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    data = load_breast_cancer(as_frame=True)
    X = data.data.values
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    images = build_synthetic_images(X_scaled)
    images = images.reshape(-1, 64, 64, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        images, y, test_size=0.2, stratify=y, random_state=42
    )

    model = make_model()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=16,
        verbose=2,
    )

    model.save(os.path.join(MODELS_DIR, "cnn_model.keras"))

    metrics = {
        "val_accuracy": float(round(history.history["val_accuracy"][-1], 4)),
        "val_auc": float(round(history.history["val_auc"][-1], 4)),
        "val_loss": float(round(history.history["val_loss"][-1], 4)),
        "train_accuracy": float(round(history.history["accuracy"][-1], 4)),
        "train_auc": float(round(history.history["auc"][-1], 4)),
        "train_loss": float(round(history.history["loss"][-1], 4)),
    }
    with open(os.path.join(PROJECT_DIR, "cnn_metrics.json"), "w", encoding="utf-8") as outfile:
        json.dump(metrics, outfile, indent=2)

    print("Image model training complete. CNN saved to models/cnn_model.keras.")


if __name__ == "__main__":
    main()
