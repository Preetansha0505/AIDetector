import spacy
from spacy.tokenizer import Tokenizer
import pandas as pd
from src.data_loader import load_data
from src.preprocess import preprocess_text
from src.perplexity import calculate_perplexity, calculate_perplexity_detailed
from src.stylometry import extract_stylometric_features
from src.paraphrase import extract_semantic_features

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Preprocessing text
# Creating a DataFrame
data = load_data("data/ai_human_detection_v1.csv")
df = pd.DataFrame(data)
sample = df.iloc[1]["text"]
preprocess_res = preprocess_text(sample)

# Calculating perplexity
perp_res = calculate_perplexity(sample)
print("Text Perplexity:", perp_res)

# Calculating stylometry ratio
stylo_res = extract_stylometric_features(sample)

# Calculating paraphrasing ratio
paraphrase_res = extract_semantic_features(preprocess_res["sentences"])

