import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

if __name__ == "__main__":
    missing = []
    expected = [
        os.path.join(MODELS_DIR, "scaler.pkl"),
        os.path.join(MODELS_DIR, "feature_names.pkl"),
        os.path.join(MODELS_DIR, "random_forest.pkl"),
        os.path.join(MODELS_DIR, "gradient_boosting.pkl"),
        os.path.join(MODELS_DIR, "svm.pkl"),
        os.path.join(MODELS_DIR, "logistic_regression.pkl"),
        os.path.join(MODELS_DIR, "knn.pkl"),
        os.path.join(MODELS_DIR, "cnn_model.keras"),
    ]
    for path in expected:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print("Missing trained artifacts. Running training scripts...")
        subprocess.check_call([sys.executable, os.path.join(PROJECT_DIR, "train_models.py")])
        subprocess.check_call([sys.executable, os.path.join(PROJECT_DIR, "train_image_model.py")])
    else:
        print("All artifacts present. Starting Flask app...")

    os.chdir(PROJECT_DIR)
    subprocess.check_call([sys.executable, os.path.join(PROJECT_DIR, "app.py")])
