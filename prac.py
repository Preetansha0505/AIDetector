# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy
from spacy.tokenizer import Tokenizer

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
# text = ("When Sebastian Thrun started working on self-driving cars at "
#         "Google in 2007, few people outside of the company took him "
#         "seriously. “I can tell you very senior CEOs of major American "
#         "car companies would shake my hand and turn away because I wasn’t "
#         "worth talking to,” said Thrun, in an interview with Recode earlier "
#         "this week.")

# text = "APIs (Application Programming Interfaces) allow different software systems to communicate with each other. Think of an API as a waiter in a restaurant—it takes your request, delivers it to the kitchen, and brings back the response. For example, when you use a weather app, it doesn't generate weather data itself. Instead, it sends a request to a weather API, which returns the current temperature and forecast."
# doc = nlp(text)

# # Analyze syntax
# print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
# print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# # Find named entities, phrases and concepts
# for entity in doc.ents:
#     print(entity.text, entity.label_)

# # Splitting sentences
# nlp.add_pipe("sentencizer")
# docx = doc
# # assert len(list(doc.sents)) == 2
# doc_lis = list(docx.sents)
# print(doc_lis)
# print(len(doc_lis))

# Tokenisation
# tokenizer = Tokenizer(nlp.vocab)
# token = tokenizer(text)

# print([token], len(token))

# from src.data_loader import load_data
# from src.preprocess import preprocess_text

# df = load_data("data/ai_human_detection_v1.csv")

# sample = df.iloc[1]["text"]

# result = preprocess_text(sample)

# print(result)

from src.perplexity import calculate_perplexity, calculate_perplexity_detailed

text1 = "APIs (Application Programming Interfaces) allow different software systems to communicate with each other. Think of an API as a waiter in a restaurant—it takes your request, delivers it to the kitchen, and brings back the response. For example, when you use a weather app, it doesn't generate weather data itself. Instead, it sends a request to a weather API, which returns the current temperature and forecast."
# text2 = "Bro I tried this and it completely broke, not even kidding."

print("Text Perplexity:", calculate_perplexity(text1))
# print("Text2 Perplexity:", calculate_perplexity(text2))