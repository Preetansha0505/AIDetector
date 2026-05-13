import numpy as np
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")


def extract_stylometric_features(text):
    doc = nlp(text)

    sentences = list(doc.sents)
    tokens = [token for token in doc if not token.is_space]

    # --- Sentence Length ---
    sentence_lengths = [len(sent) for sent in sentences]

    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0

    # --- Vocabulary Richness ---
    words = [token.text.lower() for token in tokens if token.is_alpha]
    unique_words = set(words)

    type_token_ratio = len(unique_words) / len(words) if words else 0

    # --- Stopword Ratio ---
    stopwords = [token for token in tokens if token.is_stop]
    stopword_ratio = len(stopwords) / len(tokens) if tokens else 0

    # --- POS Distribution ---
    pos_counts = Counter([token.pos_ for token in tokens])
    total_tokens = len(tokens)

    pos_distribution = {
        pos: count / total_tokens for pos, count in pos_counts.items()
    } if total_tokens else {}

    return {
        "avg_sentence_length": avg_sentence_length,
        "sentence_length_std": sentence_length_std,
        "type_token_ratio": type_token_ratio,
        "stopword_ratio": stopword_ratio,
        "pos_distribution": pos_distribution
    }
    
    
def flatten_features(features):
    flat = {}

    for key, value in features.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                flat[f"{key}_{sub_key}"] = sub_val
        else:
            flat[key] = value

    return flat