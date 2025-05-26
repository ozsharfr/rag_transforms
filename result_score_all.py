def calc_score_from_llm(retrieved_documents: list, question: str, llm) -> list[int]:
    """
    Score multiple documents in a single LLM call for efficiency
    """
    if not retrieved_documents:
        return []
    
    # Build the batch prompt
    prompt = f"""You are a highly reliable medical assistant.
    Rate the relevance of each document below to the query on a scale of 1-10.
    Consider the specific intent of the query.

    Query: {question}

    Please provide scores in this exact format: [score1, score2, score3, ...]
    where each score is a number between 1 and 10.

    """
    
    # Add each document with clear numbering
    for i, doc in enumerate(retrieved_documents, 1):
        prompt += f"\nDocument {i}:\n{doc}\n"
    
    prompt += "\nRelevance Scores (format: [score1, score2, score3, ...]):"
    
    # Get response from LLM
    response = llm.invoke(prompt)
    
    # Extract scores from response
    scores = extract_scores_from_batch_response(response, len(retrieved_documents))
    
    return scores

def extract_scores_from_batch_response(response: str, num_docs: int) -> list[int]:
    """
    Extract numerical scores from LLM batch response
    """
    import re
    
    # Try to find list format like [7, 8, 6, 9, 5]
    list_match = re.search(r'\[([0-9,\s]+)\]', response)
    if list_match:
        numbers_str = list_match.group(1)
        try:
            scores = [int(x.strip()) for x in numbers_str.split(',')]
            if len(scores) == num_docs:
                return [max(1, min(10, score)) for score in scores]  # Clamp to 1-10
        except ValueError:
            pass
    
    # Fallback: find individual numbers in order
    numbers = re.findall(r'\b([1-9]|10)\b', response)
    if len(numbers) >= num_docs:
        scores = [int(num) for num in numbers[:num_docs]]
        return scores
    
    # Last resort: default scores
    print(f"Warning: Could not parse scores from response. Using default scores.")
    return [5] * num_docs  # Default to middle score

# Alternative version if you want to keep your existing extract function
def calc_score_from_llm_alternative(retrieved_documents: list, question: str, llm) -> list[int]:
    """
    Alternative version using your existing get_num function
    """
    if not retrieved_documents:
        return []
    
    prompt = f"""You are a highly reliable medical assistant.
Rate the relevance of each document below to the query on a scale of 1-10.
Consider the specific intent of the query.

Query: {question}

"""
    
    for i, doc in enumerate(retrieved_documents, 1):
        prompt += f"\nDocument {i}:\n{doc}\n"
    
    prompt += """
Please provide scores in this format:
Document 1: X out of 10
Document 2: Y out of 10
Document 3: Z out of 10
(and so on...)

Relevance Scores:"""
    
    response = llm.invoke(prompt)
    
    # Use your existing get_num function to extract each score
    lines = response.split('\n')
    scores = []
    
    for line in lines:
        if 'out of 10' in line.lower():
            score = get_num(line)  # Your existing function
            if score is not None:
                scores.append(score)
    
    # Ensure we have the right number of scores
    while len(scores) < len(retrieved_documents):
        scores.append(5)  # Default score
    
    return scores[:len(retrieved_documents)]