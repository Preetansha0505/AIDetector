from src.data_loader import load_data
from src.preprocess import preprocess_text
from src.perplexity import calculate_perplexity, calculate_perplexity_detailed
from src.stylometry import extract_stylometric_features
from src.paraphrase import extract_semantic_features

def extract_features(text):
    # Preprocessing text
    preprocess_res = preprocess_text(text)

    # Calculating perplexity
    perp_res = calculate_perplexity(text)
    print("Text Perplexity:", perp_res)

    # Calculating stylometry ratio
    stylo_res = extract_stylometric_features(text)

    # Calculating paraphrasing ratio
    paraphrase_res = extract_semantic_features(preprocess_res["sentences"])
    
    features = {
        **stylo_res,
        **paraphrase_res,
        "perplexity": perp_res,
        "num_sentences": preprocess_res["num_sentences"],
        "num_tokens": preprocess_res["num_tokens"]
    }
    
    return features

