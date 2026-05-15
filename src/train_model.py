import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train():
    df = pd.read_csv("data/processed/features.csv")

    y = df["label"]
    X = df.drop(columns=["label"])
    joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    joblib.dump(model, "model/ai_detector.pkl")
    preds = model.predict(X_test)

    print(classification_report(y_test, preds))

    return model


if __name__ == "__main__":
    train()