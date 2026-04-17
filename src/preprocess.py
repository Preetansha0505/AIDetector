import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text)
    
    sentences = [sent.text.strip() for sent in doc.sents]
    tokens = [token.text for token in doc if not token.is_space]
    
    return {
        "sentences": sentences,
        "tokens": tokens,
        "num_sentences": len(sentences),
        "num_tokens": len(tokens)
    }
    
def is_valid_text(text):
    if not isinstance(text, str):
        return False
    
    # too short → useless
    if len(text.split()) < 30:
        return False
    
    # filter common error patterns
    blacklist = [
        "Error:",
        "404",
        "Not Found",
        "Traceback",
        "Exception",
        "api.",
        "http"
    ]
    
    for word in blacklist:
        if word.lower() in text.lower():
            return False
    
    return True