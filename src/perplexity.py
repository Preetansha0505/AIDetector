from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math

# Load once (IMPORTANT: not inside function)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

model.eval()


def calculate_perplexity(text):
    encodings = tokenizer(text, return_tensors="pt")
    
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = torch.exp(loss).item()

    return perplexity

def calculate_perplexity_detailed(text):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    perplexity = torch.exp(loss).item()

    return {
        "avg_perplexity": perplexity,
        "num_tokens": input_ids.shape[1]
    }