from sentence_transformers import SentenceTransformer, util

def compare_answers(llm_response:str, reference_texts : list[str] , k:int = 5) -> tuple[list[str], list[float]]:
    best_chunks = []
    # Load BioBERT model
    model = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")
    #model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    # Compute embeddings
    response_embedding = model.encode(llm_response, convert_to_tensor=True)
    reference_embeddings = model.encode(reference_texts, convert_to_tensor=True)

    # Calculate cosine similarities
    cosine_scores = util.cos_sim(response_embedding, reference_embeddings)
    cosine_scores = cosine_scores[0].numpy()  # Get the first row of the cosine similarity matrix

    best_ixs = cosine_scores.argsort()[-k:]
    for match_idx in best_ixs:
        best_chunks.append(reference_texts[match_idx]) #, Score: {cosine_scores[match_idx]}")
        print (f"score: {cosine_scores[match_idx]}")

    return best_chunks , cosine_scores