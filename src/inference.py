import joblib
import pandas as pd

from src.feature_pipeline import extract_features

# Load model
model = joblib.load("model/ai_detector.pkl")

# Load training feature schema
feature_columns = joblib.load("model/feature_columns.pkl")


def predict_text(text):

    feats = extract_features(text)

    X = pd.DataFrame([feats])

    # --- Add missing columns ---
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0

    # --- Remove unexpected extra columns ---
    X = X[feature_columns]

    probability = model.predict_proba(X)[0]

    predicted_class = model.classes_[probability.argmax()]
    confidence = probability.max()

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 3),
        "probabilities": {
            cls: round(prob, 3)
            for cls, prob in zip(model.classes_, probability)
        }
    }