import json
import os
from collections import OrderedDict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)


def build_models():
    return OrderedDict(
        [
            ("Random Forest", RandomForestClassifier(n_estimators=180, random_state=42)),
            ("Gradient Boosting", GradientBoostingClassifier(n_estimators=120, random_state=42)),
            ("SVM", SVC(probability=True, kernel="rbf", random_state=42)),
            ("Logistic Regression", LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)),
            ("KNN", KNeighborsClassifier(n_neighbors=7)),
        ]
    )


def main():
    data = load_breast_cancer(as_frame=True)
    X = data.data.values
    y = data.target
    feature_names = data.feature_names.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()
    results = {"models": []}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        if hasattr(model, "fit"):
            model.fit(X_train_scaled, y_train)

        scores = cross_validate(
            model,
            X_train_scaled,
            y_train,
            cv=cv,
            scoring=["accuracy", "roc_auc", "f1"],
            return_train_score=False,
            n_jobs=1,
        )

        entry = {
            "name": name,
            "accuracy": float(round(scores["test_accuracy"].mean(), 4)),
            "roc_auc": float(round(scores["test_roc_auc"].mean(), 4)),
            "f1": float(round(scores["test_f1"].mean(), 4)),
            "parameters": model.get_params(),
        }
        results["models"].append(entry)

        model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    with open(os.path.join(PROJECT_DIR, "model_results.json"), "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2)

    print("Training complete. Models and metadata saved to models/ and model_results.json.")


if __name__ == "__main__":
    main()
