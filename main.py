from langchain_ollama import OllamaLLM
from transformers_compare import compare_answers
import time 

from doc_specific_parser import filter_conflict_lines, filter_author_like_lines
from text_split import split_into_chunks
from prompts_formatted  import format_prompt_initial, format_rag_prompt
from result_score import calc_score_from_llm

from dotenv import load_dotenv
load_dotenv()


if __name__ == '__main__':

    # File path for the input document
    FILE_PATH = 'abstracts_park.txt'
    # Read the document and split it into chunks
    t  = time.time()
    with open(FILE_PATH, "r", encoding='utf-8') as file:
        document_text = file.read()
        document_text = filter_conflict_lines(document_text)
        document_text = filter_author_like_lines(document_text)

    for chunk_size in [400,600, 800]:
        
        chunks = split_into_chunks(document_text, chunk_size=chunk_size , chunk_overlap=300)

        chunks = chunks[:] ### limit chaunks number (or fast screen, by simpler model) for reasonable runtime

        print (f"Chunk size: {chunk_size} , number of chunks: {len(chunks)}")

        query = "What are the possible parkinson treatments"

        result_llama = format_prompt_initial(query=query, llm = OllamaLLM(model= 'llama3'))

        retrieved_documents , scrs = compare_answers(llm_response = result_llama , reference_texts =chunks)
        scored_docs = calc_score_from_llm(retrieved_documents, question=query, llm = OllamaLLM(model= 'llama3'))
        print (f"Scored docs: {scored_docs}")
        filterd_documents = [ doc for doc, score in zip(retrieved_documents, scored_docs) if score >= 5]
        if len(filterd_documents) == 0:
            print ("No relevant documents found")
        else:
            final_result = format_rag_prompt(filterd_documents, query, llm = OllamaLLM(model= 'llama3'))
            # Check the below:
            final_answer_score = calc_score_from_llm([final_result], question=query, llm = OllamaLLM(model= 'llama3'))
            print (f"Final answer score: {final_answer_score}")
            print (f"Final answer: {final_result}")
        print (f" Time taken: {time.time() - t:.2f} seconds")
        print ("=========================")
 