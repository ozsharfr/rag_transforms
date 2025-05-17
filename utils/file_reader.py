
def read_single_file(FILE_PATH: str) -> str:
    with open(FILE_PATH, "r", encoding='utf-8') as file:
        document_text = file.read()
        return document_text
    return None

    