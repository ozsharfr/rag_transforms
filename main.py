import os
import time
import logging
from typing import Optional, List, Tuple, Union

from config import Config
from langchain_ollama import OllamaLLM
from transformers_embed import nearest_sentences, embed_text
from text_split import split_into_chunks
from prompts_formatted import format_prompt_initial, format_rag_prompt
from result_score_all import calc_score_from_llm
from utils.doc_parser import filter_conflict_lines, filter_author_like_lines
from utils.logger import setup_logger
from utils.file_reader import read_single_file
from test_chroma import run_chroma, nearest_to_q

# Disable noisy transformer output
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Configure logging
logger = setup_logger(log_file=Config.LOG_FILE, log_level=Config.LOG_LEVEL)

# Global cache for preprocessed chunks
_cached_chunks: Optional[List[str]] = None

def read_and_clean_document() -> Optional[str]:
    """Read and clean the document from the file system."""
    logger.info(f"Reading and processing document {Config.FILE_PATH}...")
    document_text = read_single_file(Config.FILE_PATH)

    if not document_text:
        logger.error("Document not found or empty.")
        return None

    document_text = filter_conflict_lines(document_text)
    document_text = filter_author_like_lines(document_text)
    return document_text

def process_document(document_text: str) -> List[str]:
    """Split the document into chunks, using cache if available."""
    global _cached_chunks

    if _cached_chunks is not None:
        logger.info(f"Using cached chunks: {_cached_chunks and len(_cached_chunks)} chunks")
        return _cached_chunks

    chunks = split_into_chunks(document_text, chunk_size=Config.CHUNK_SIZE, chunk_overlap=300)
    _cached_chunks = chunks
    logger.info(f"Chunk size: {Config.CHUNK_SIZE}, number of chunks: {len(chunks)}")
    return chunks

def get_reference_embeddings(chunks: List[str], cached_embeddings: Optional[List] = None) -> List:
    """Return cached or new reference embeddings."""
    if cached_embeddings:
        logger.info("Using cached embeddings - skipping embedding step")
        return cached_embeddings

    logger.info("Creating new embeddings...")
    if Config.BOOL_CHROMADB:
        return run_chroma(chunks)
    else:
        return embed_text(text=chunks)

def retrieve_documents(modified_query: str, reference_embeddings: List, chunks: List[str]) -> List[str]:
    """Retrieve the top relevant documents using the selected DB mode."""
    if Config.BOOL_CHROMADB:
        return nearest_to_q(modified_query, reference_embeddings, n_results=Config.RETRIEVE_TOP_K)

    retrieved_documents, _ = nearest_sentences(
        llm_response=modified_query,
        reference_texts=chunks,
        reference_embeddings=reference_embeddings,
        k=Config.RETRIEVE_TOP_K
    )
    return retrieved_documents

def build_final_answer(
    retrieved_docs: List[str],
    query: str,
    llm_model: OllamaLLM,
    relevance_scores: List[Union[int, float]],
    start_time: float
) -> str:
    """Format and return the final RAG result."""
    logger.info(f"Scored docs: {relevance_scores}")
    relevant_docs = [doc for doc, score in zip(retrieved_docs, relevance_scores) if score >= Config.MIN_RELEVANCE_SCORE]

    if not relevant_docs:
        logger.info("No relevant documents found")
        return "No relevant documents found for the given query."

    final_result = format_rag_prompt(relevant_docs, query, llm=llm_model)
    logger.info(f"Final answer: {final_result}")
    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds")
    logger.info("=========================")
    return final_result

def main(
    query: str = "What are the possible Parkinson treatments",
    log_stream = None,
    cached_embeddings: Optional[List] = None
) -> Tuple[Optional[List], str]:
    """
    Run the complete RAG pipeline: read, chunk, embed, retrieve, and generate.

    Args:
        query (str): User query.
        log_stream (io.StringIO, optional): Stream for capturing logs.
        cached_embeddings (list, optional): Cached reference embeddings.

    Returns:
        tuple: (reference_embeddings, final_answer)
    """
    global _cached_chunks
    start_time = time.time()

    document_text = read_and_clean_document()
    if not document_text:
        return None, "Failed to load document."

    chunks = process_document(document_text)
    llm_model = OllamaLLM(model=Config.MODEL_NAME, base_url=Config.OLLAMA_HOST)

    reference_embeddings = get_reference_embeddings(chunks, cached_embeddings)

    logger.info("Get enriched query from LLM (consider removing)")
    modified_query = format_prompt_initial(query=query, llm=llm_model)
    logger.info(f"Modified query: {modified_query}")

    retrieved_docs = retrieve_documents(modified_query, reference_embeddings, chunks)

    relevance_scores = [10] * len(retrieved_docs)  # Placeholder for scoring logic

    final_answer = build_final_answer(retrieved_docs, query, llm_model, relevance_scores, start_time)
    return reference_embeddings, final_answer

if __name__ == "__main__":
    main()
