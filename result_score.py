import re
from langchain.prompts import PromptTemplate

def get_num(score_text):
    pattern = r'(\d+)\s+out of'
    ee = re.search(pattern, score_text)
    if ee is not None:
        return int(score_text[ee.regs[-1][0]: ee.regs[-1][1]])
    else:
        return None


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