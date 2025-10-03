import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def embed_text(text: str) -> np.ndarray:
    """
    Fake embedding function (you can replace with real model).
    Currently splits text length into vector form.
    """
    return np.array([len(text), sum(ord(c) for c in text) % 1000])


def find_top_matches(query_vector, vectors, top_k=5):
    """Find top-k similar vectors using cosine similarity."""
    query_np = np.array([query_vector])      # (1, d)
    vectors_np = np.array(vectors)           # (n, d)

    sims = cosine_similarity(query_np, vectors_np)[0]  # similarity scores
    top_indices = sims.argsort()[-top_k:][::-1]        # top matches
    return [(i, float(sims[i])) for i in top_indices]
