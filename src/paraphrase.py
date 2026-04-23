from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_semantic_features(sentences):
    if len(sentences) < 2:
        return {
            "avg_sentence_similarity": 0,
            "max_similarity": 0,
            "redundancy_score": 0
        }

    embeddings = model.encode(sentences)

    sim_matrix = cosine_similarity(embeddings)

    # remove diagonal (self-similarity)
    sim_scores = []
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            sim_scores.append(sim_matrix[i][j])

    avg_similarity = np.mean(sim_scores)
    max_similarity = np.max(sim_scores)

    # redundancy: % of pairs above threshold
    threshold = 0.8
    redundant_pairs = [s for s in sim_scores if s > threshold]
    redundancy_score = len(redundant_pairs) / len(sim_scores)

    return {
        "avg_sentence_similarity": avg_similarity,
        "max_similarity": max_similarity,
        "redundancy_score": redundancy_score
    }