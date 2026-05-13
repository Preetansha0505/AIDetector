import pandas as pd
from tqdm import tqdm
from src.data_loader import load_data
from src.feature_pipeline import extract_features

def build_feature_dataset(input_path, output_path):
    df = load_data(input_path)

    feature_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        text = row["text"]
        label = row["human_or_ai"]

        feats = extract_features(text)
        clean_feats = {}

        for k, v in feats.items():
            if isinstance(v, (int, float)):
                clean_feats[k] = v
        
        clean_feats["label"] = label
        feature_rows.append(clean_feats)

    feature_df = pd.DataFrame(feature_rows)
    feature_df.fillna(0, inplace=True)

    feature_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    build_feature_dataset("data/ai_human_detection_v1.csv", "data/processed/features.csv")