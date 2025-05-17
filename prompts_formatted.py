
from langchain.prompts import PromptTemplate

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