import torch
import pickle
import faiss

def generate_translation(word_embeddings):
    word_embedding_keys, word_embedding_values = zip(*word_embeddings.items())
    faiss_index = faiss.IndexFlatL2(word)

    def vec2word(embeddings):
        indices = faiss_index.search(words, 5)[1]
        return [word_embedding_keys[i] for i in indices]

    return vec2word
