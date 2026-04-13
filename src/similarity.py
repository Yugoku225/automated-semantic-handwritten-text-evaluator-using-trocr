import logging
from sentence_transformers import SentenceTransformer, util

# Optimize initialization by loading the model globally
# "all-MiniLM-L6-v2" is a lightweight, fast, and effective model for semantic similarity.
print("[*] Loading Sentence-BERT Model (this might take a few seconds on first run)...")
# Suppress transformers logging for cleaner output
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(student_text, model_text):
    """
    Computes the cosine similarity between a student's answer and the model answer.
    Returns a float between 0.0 and 1.0 representing similarity.
    """
    print("    -> Encoding sentences to dense vectors...")
    
    # Compute embeddings for both texts
    embeddings1 = model.encode(student_text, convert_to_tensor=True)
    embeddings2 = model.encode(model_text, convert_to_tensor=True)
    
    # Compute cosine similarity
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    
    # The output of cos_sim is a tensor matrix, we want the single float value
    score = cosine_scores[0][0].item()
    
    return score
