from sentence_transformers import SentenceTransformer, util
from config import Config

model = SentenceTransformer(Config.TRANSFORMER_MODEL)

def embed_text(text: str) -> list[float]:
    """
    Embed a given text using the SentenceTransformer model.
    
    Args:
        text (str): The text to embed.
        
    Returns:
        list[float]: The embedding of the text.
    """
    
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding

def nearest_sentences(llm_response:str, reference_texts : list[str] , k:int = 5, reference_embeddings = None ) -> tuple[list[str], list[float]]:
    best_chunks = []
    # Load BioBERT model

    response_embedding = embed_text(llm_response)

    # Calculate cosine similarities
    cosine_scores = util.cos_sim(response_embedding, reference_embeddings)
    cosine_scores = cosine_scores[0].numpy()  # Get the first row of the cosine similarity matrix

    best_ixs = cosine_scores.argsort()[-k:]
    for match_idx in best_ixs:
        best_chunks.append(reference_texts[match_idx]) #, Score: {cosine_scores[match_idx]}")
        print (f"score: {cosine_scores[match_idx]}")

    return best_chunks , cosine_scores