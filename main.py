from langchain_ollama import OllamaLLM
from transformers_embed import nearest_sentences
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

def main(query ="What are the possible parkinson treatments",  log_stream=None):
    
    # Read the document and split it into chunks
    t  = time.time()
    document_text = read_single_file(Config.FILE_PATH)
    if document_text is not None:               
        document_text = filter_conflict_lines(document_text)
        document_text = filter_author_like_lines(document_text)
    else:
        logger.error("Document not found or empty.")
        return
    
    llm_model = OllamaLLM(model=Config.MODEL_NAME, base_url=Config.OLLAMA_HOST)
    #llm_model = OllamaLLM(model=Config.MODEL_NAME)

    chunk_size = Config.CHUNK_SIZE
        
    chunks = split_into_chunks(document_text, chunk_size=chunk_size , chunk_overlap=300)

    chunks = chunks[:] ### limit chaunks number (or fast screen, by simpler model) for reasonable runtime

    logger.info(f"Chunk size: {chunk_size} , number of chunks: {len(chunks)}")

    result_llama = format_prompt_initial(query=query, llm = llm_model)

    retrieved_documents , scrs = nearest_sentences(llm_response = result_llama , reference_texts =chunks)
    scored_docs = calc_score_from_llm(retrieved_documents, question=query, llm = llm_model)
    logger.info (f"Scored docs: {scored_docs}")
    filterd_documents = [ doc for doc, score in zip(retrieved_documents, scored_docs) if score >= Config.MIN_RELEVANCE_SCORE]
    if len(filterd_documents) == 0:
        logger.info ("No relevant documents found")
    else:
        final_result = format_rag_prompt(filterd_documents, query, llm = llm_model)
        # Check the below:
        final_answer_score = calc_score_from_llm([final_result], question=query, llm = llm_model)
        logger.info (f"Final answer score: {final_answer_score}")
        logger.info (f"Final answer: {final_result}")
        print (f"Final answer: {final_result}")
    logger.info (f" Time taken: {(time.time() - t):.2f} seconds")
    logger.info ("=========================")

if __name__ == "__main__":
    main()
 