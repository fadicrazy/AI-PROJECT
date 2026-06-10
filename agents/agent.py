import random


class CancerAgent:
    def __init__(self):
        self.intro = (
            "This assistant provides research-oriented guidance based on the Breast Cancer Wisconsin dataset "
            "and model predictions. All insights are not medical advice and are intended for educational use only."
        )
        self.recommendations = {
            "Low": [
                "Maintain regular screenings and healthy lifestyle choices.",
                "Review the features with your physician if any values are borderline.",
                "Continue monitoring and use the model output for awareness.",
            ],
            "Moderate": [
                "Seek follow-up imaging or clinical consultation.",
                "Compare the flagged metrics with historical records.",
                "Schedule a specialist review for further evaluation.",
            ],
            "High": [
                "Prioritize prompt diagnostic follow-up and clinical assessment.",
                "Discuss biopsy or advanced imaging options with your healthcare provider.",
                "Use this result as an early warning signal, not a diagnosis.",
            ],
        }

    def generate_report(self, label, probability, risk, features):
        verdict = (
            "The prediction suggests a benign profile." if label == "Benign"
            else "The prediction indicates a malignant profile."
        )
        reasons = []
        if features:
            sorted_features = sorted(features.items(), key=lambda item: abs(item[1]), reverse=True)
            for key, value in sorted_features[:4]:
                reasons.append(f"{key}: {round(value, 2)}")
        else:
            reasons.append("Image-based classification using the trained CNN model.")

        return {
            "summary": verdict,
            "risk_level": risk,
            "confidence": float(round(probability * 100, 2)),
            "key_reasons": reasons,
            "recommendations": self.recommendations.get(risk, []),
        }

    def generate_report_for_sample(self, sample_result):
        labels = [prediction["label"] for prediction in sample_result["predictions"]]
        majority = max(set(labels), key=labels.count)
        return {
            "summary": f"Sample predicted as {majority} by most models.",
            "risk_level": "High" if majority == "Malignant" else "Low",
            "notes": [
                "This demo uses five different classifiers for tabular breast cancer data.",
                "Compare the probability spread to understand model agreement.",
            ],
        }

    def chat(self, question, context):
        query = question.strip().lower()
        if not query:
            return "Please ask a question about the dataset, models, or predictions."

        if "dataset" in query or "samples" in query or "features" in query:
            return (
                "The Wisconsin Breast Cancer dataset includes 569 samples and 30 numeric features, "
                "with a binary target for malignant and benign tumors. It is ideal for classification research."
            )

        if "random forest" in query:
            return (
                "Random Forest is an ensemble classifier that aggregates decision trees. "
                "It tends to be robust for the breast cancer dataset and handles feature interactions well."
            )

        if "svm" in query or "support vector" in query:
            return (
                "SVM is effective for medium-sized datasets such as this one. "
                "It can separate classes using a soft margin and kernel mapping."
            )

        if "cnn" in query or "image" in query:
            return (
                "The CNN model accepts 64x64 grayscale images and predicts tumor likelihood from synthetic imaging patterns. "
                "Treatment decisions should not rely solely on this result."
            )

        if "recommend" in query or "next step" in query or "risk" in query:
            return (
                "Use the prediction as a screening signal. Low risk suggests monitoring, moderate risk suggests follow-up imaging, "
                "and high risk suggests prompt clinical evaluation."
            )

        if "confidence" in query or "probability" in query:
            return (
                "Model confidence is based on predicted probability. Higher scores mean greater certainty in the predicted class, "
                "but they are not a substitute for clinical diagnosis."
            )

        if "feature importance" in query or "important features" in query:
            return (
                "Key predictive features include mean radius, mean texture, mean perimeter, and worst area. "
                "These metrics often contribute strongly to discrimination between malignant and benign tumors."
            )

        quick_answers = [
            "Please tell me more about the dataset or which model you want to discuss.",
            "I can compare model metrics, explain the AI risk levels, or recommend next steps for evaluation.",
        ]
        return random.choice(quick_answers)
