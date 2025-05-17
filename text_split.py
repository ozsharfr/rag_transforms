from langchain.text_splitter import RecursiveCharacterTextSplitter


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