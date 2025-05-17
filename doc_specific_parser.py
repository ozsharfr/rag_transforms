import re

def filter_conflict_lines(document_text  : str) -> str:
    """
    Filters out lines that start with 'Conflict of interest' or any other format from the document text.
    Args:
        document_text (str): The input document text.
    Returns:
        str: The filtered document text.
    """ 
    
    lines = document_text.split('\n\n')
    lines_filtered = [l for l in lines if not l.lower().startswith('conflict')]
    return '\n'.join(lines_filtered)

def filter_author_like_lines(document_text: str) -> str:
    """
    Filters out lines that contain author-like patterns (e.g., "(1)", "(2)") from the document text.    
    Args:
        document_text (str): The input document text.
    Returns:
        str: The filtered document text.
    """
    pattern = r'\(\d+\)'
    lines_filtered = [line for line in document_text.split('\n') if not bool(re.search(pattern,line))]
    return '\n'.join(lines_filtered)


