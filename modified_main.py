from langchain_ollama import OllamaLLM
from transformers_embed import nearest_sentences , embed_text
import time 
from text_split import split_into_chunks
from prompts_formatted  import format_prompt_initial, format_rag_prompt
from result_score_all import calc_score_from_llm

from utils.doc_parser import filter_conflict_lines, filter_author_like_lines
from utils.logger import setup_logger
from utils.file_reader import read_single_file

from config import Config 
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

logger = setup_logger(log_file=Config.LOG_FILE, log_level=Config.LOG_LEVEL)

# Global cache for chunks (so we don't re-process the document)
_cached_chunks = None

def final_prompt_based_on_retrieved_docs(retrieved_documents, query, llm_model, scored_docs, t):
    """
    Formats the final prompt based on the retrieved documents and the query.
    
    Args:
        retrieved_documents (list): List of retrieved documents.
        query (str): The original query.
        llm_model: The language model to use for formatting.
        scored_docs: Relevance scores for documents.
        t: Start time for timing.
    
    Returns:
        str: The final answer or None if no relevant documents.
    """
    logger.info (f"Scored docs: {scored_docs}")
    filterd_documents = [ doc for doc, score in zip(retrieved_documents, scored_docs) if score >= Config.MIN_RELEVANCE_SCORE]
    
    if len(filterd_documents) == 0:
        logger.info ("No relevant documents found")
        final_result = "No relevant documents found for the given query."
    else:
        final_result = format_rag_prompt(filterd_documents, query, llm = llm_model)
        # Check the below:
        #final_answer_score = calc_score_from_llm([final_result], question=query, llm = llm_model)
        #logger.info (f"Final answer score: {final_answer_score}")
        logger.info (f"Final answer: {final_result}")
    
    logger.info (f" Time taken: {(time.time() - t):.2f} seconds")
    logger.info ("=========================")
    return final_result
    
def main(query="What are the possible parkinson treatments", log_stream=None, cached_embeddings=None):
    global _cached_chunks
    
    # Start timing
    t = time.time()
    
    # Use cached chunks if available, otherwise process document
    if _cached_chunks is None:
        logger.info("Reading and processing document...")
        document_text = read_single_file(Config.FILE_PATH)
        if document_text is not None:               
            document_text = filter_conflict_lines(document_text)
            document_text = filter_author_like_lines(document_text)
        else:
            logger.error("Document not found or empty.")
            return None
        
        chunk_size = Config.CHUNK_SIZE
        chunks = split_into_chunks(document_text, chunk_size=chunk_size, chunk_overlap=300)
        chunks = chunks[:100]  # limit chunks number for reasonable runtime
        _cached_chunks = chunks
        logger.info(f"Chunk size: {chunk_size}, number of chunks: {len(chunks)}")
    else:
        chunks = _cached_chunks
        logger.info(f"Using cached chunks: {len(chunks)} chunks")
    
    # Initialize LLM
    llm_model = OllamaLLM(model=Config.MODEL_NAME, base_url=Config.OLLAMA_HOST)
    
    # Use cached embeddings if available, otherwise create new ones
    if cached_embeddings is not None:
        logger.info("Using cached embeddings - skipping embedding step")
        reference_embeddings = cached_embeddings
    else:
        logger.info("Creating new embeddings...")
        reference_embeddings = embed_text(text=chunks)
    
    # Get initial LLM response
    result_llama = format_prompt_initial(query=query, llm=llm_model)
    
    # Retrieve relevant documents
    retrieved_documents, scrs = nearest_sentences(
        llm_response=result_llama, 
        reference_texts=chunks,
        reference_embeddings=reference_embeddings
    )
    
    # Score the retrieved documents
    scored_docs = calc_score_from_llm(retrieved_documents, question=query, llm=llm_model)
    
    # Generate final answer
    final_answer = final_prompt_based_on_retrieved_docs(retrieved_documents, query, llm_model, scored_docs, t)
    
    # Return embeddings for caching
    return reference_embeddings , final_answer

if __name__ == "__main__":
    main()