from src.feature_pipeline import extract_features

def predict(text, model):
    feats = extract_features(text)

    import pandas as pd
    X = pd.DataFrame([feats])

    prob = model.predict_proba(X)[0][1]

    return {
        "ai_probability": prob,
        "prediction": "AI" if prob > 0.5 else "Human",
        "confidence": "High" if prob > 0.75 or prob < 0.25 else "Medium"
    }