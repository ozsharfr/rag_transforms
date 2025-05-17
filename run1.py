from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import embeddings , OllamaLLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import PromptTemplate

from transformers_compare import compare_answers
import numpy as np
import re
import time 

from dotenv import load_dotenv

load_dotenv()

 

embeddings = OllamaEmbeddings(
    model="llama3",
)

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

def retrieve_dynamic(chunks, query, embeddings, k=5):
    """
    Retrieve the most relevant chunks based on query similarity using embeddings.
    Args:
        chunks (list): List of document chunks.
        query (str): The input query.
        embeddings (OllamaEmbeddings): The embedding model.
        k (int): Number of top chunks to retrieve.

    Returns:
        list: Top-k most relevant chunks.
    """
    # Embed the query and chunks
    query_embedding = embeddings.embed_query(query)
    chunk_embeddings = embeddings.embed_documents(chunks)

    # Calculate cosine similarity
    scores = cosine_similarity(np.array(query_embedding).reshape(1,-1), np.array(chunk_embeddings))[0]
    # Sort by highest similarity
    top_k_indices = np.argsort(scores)[-k:][::-1]
    top_k_chunks = [chunks[i] for i in top_k_indices]
    top_k_scores = [scores[i] for i in top_k_indices]

    # Print scores for debugging (optional)
    for i, (chunk, score) in enumerate(zip(top_k_chunks, top_k_scores)):
        print(f"Chunk {i+1}: Score = {score:.4f}\n{chunk}\n")

    return top_k_chunks

def filter_conflict_lines(document_text):
    lines = document_text.split('\n\n')
    lines_filtered = [l for l in lines if not l.lower().startswith('conflict')]
    return '\n'.join(lines_filtered)

def filter_author_like_lines(document_text):
    pattern = r'\(\d+\)'
    lines_filtered = [line for line in document_text.split('\n') if not bool(re.search(pattern,line))]
    return '\n'.join(lines_filtered)

def get_num(score_text):
    pattern = r'(\d+)\s+out of'
    ee = re.search(pattern, score_text)
    if ee is not None:
        return int(score_text[ee.regs[-1][0]: ee.regs[-1][1]])
    else:
        return None

 

def split_into_chunks(text: str, chunk_size: int = 800 , chunk_overlap : int = 100) -> list[str]:
    """
    Split a given text into chunks of specified size using RecursiveCharacterTextSplitter.

    Args:
        text (str): The input text to be split into chunks.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 800.
 
    Returns:
        list[str]: A list of text chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    documents = text_splitter.create_documents([text])
    return [document.page_content for document in documents]

 

def calc_score_from_llm(retrieved_documents : list, question : str , llm) -> list[int]:
    prompt_template = PromptTemplate(
            input_variables=["query", "doc"],
            template="""You are a highly reliable medical assistant.
            On a scale of 1-10, rate the relevance of the following document to the query. 
            Consider the specific intent of the query.
            Relevance score should be in format of: 'X out of 10', where X is a number between 1 and 10.
            Query: {query}
            Document: {doc}
            Relevance Score:"""
        )

    extract_score = lambda x: get_num(x)
    llm_chain = prompt_template | llm | extract_score
    scored_docs = []

    for doc in retrieved_documents:
        input_data = {"query": question, "doc": doc}
        score = llm_chain.invoke(input_data)
        scored_docs.append(score)
    return scored_docs

def format_prompt_initial(query: str, llm ) -> str:

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=
        """You are a highly reliable medical assistant.\n
        User Query:
        {query}
        Task:
        1. Include the query at the beginning of your answer
        2. Afterwards, add your answer - as brief as possible, yet as informative as possible.
        Answer:"""
    )
    
    llm_chain = prompt_template | llm 

    input_data = {"query": query}
    result = llm_chain.invoke(input_data)

    return result

def format_rag_prompt(retrieved_docs: list[str], query: str, llm ) -> str:
    """
    Formats the RAG prompt using the provided retrieved documents and query.

    Args:
        retrieved_docs (List[str]): List of retrieved documents.
        query (str): User query.

    Returns:
        str: The formatted prompt.
    """
    # Construct the context section
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
    
    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=
        """You are a highly reliable medical assistant.\n
        You have been provided with the following context information from trusted sources.\n\n
        Context:
        {context}
        User Query:
        {query}
        Task:
        1. Answer the query **based solely on the provided context**.
        2. **If information is not available in the context, clearly state: 'No relevant information found in the context.'**
        Answer:"""
    )
    
    llm_chain = prompt_template | llm 

    input_data = {"query": query, "context": context}
    result = llm_chain.invoke(input_data)

    return result

 

if __name__ == '__main__':

    # File path for the input document
    FILE_PATH = 'abstracts_park.txt'
    # Read the document and split it into chunks
    t  = time.time()
    with open(FILE_PATH, "r", encoding='utf-8') as file:
        document_text = file.read()
        document_text = filter_conflict_lines(document_text)
        document_text = filter_author_like_lines(document_text)

    for chunk_size in [800]:
        
        chunks = split_into_chunks(document_text, chunk_size=chunk_size , chunk_overlap=chunk_size//2)

        chunks = chunks[:] ### for reasonable runtime
        print (f"Chunk size: {chunk_size} , number of chunks: {len(chunks)}")

        query = "What are the possible parkinson treatments"
        
        #retrieved_documents = retrieve_dynamic(chunks, query, embeddings, k=5)
        #vectorstore = InMemoryVectorStore.from_texts(
        #    chunks,
        #    embedding=embeddings,
        #)

        #retriever = vectorstore.as_retriever(k=3)
        #retrieved_documents = retriever.invoke(question)
        #retrieved_documents = [doc.page_content for doc in retrieved_documents]

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
 