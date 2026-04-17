import pandas as pd
from src.preprocess import is_valid_text

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    df = df[df["text"].apply(is_valid_text)]
    return df

def split_data(df, test_size=0.2):
    from sklearn.model_selection import train_test_split
    
    X = df["text"]
    y = df["label"]
    
    return train_test_split(X, y, test_size=test_size, random_state=42)

