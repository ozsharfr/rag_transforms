import chromadb
from sentence_transformers import SentenceTransformer
from config import Config
from utils.logger import setup_logger

model = SentenceTransformer(Config.TRANSFORMER_MODEL)  # Or any other model
client = chromadb.Client()

logger = setup_logger(log_file=Config.LOG_FILE, log_level=Config.LOG_LEVEL)

# Initialize ChromaDB client and collection
def run_chroma(docs):
    collection = client.create_collection('docs',
                            metadata={"hnsw:space": "cosine"})
    # Embed and add documents to ChromaDB
    embeddings = model.encode(docs).tolist()
    collection.add(
        documents=docs,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(docs))]
    )
    return collection


# Your search query
def nearest_to_q(query , collection , n_results):
    # Embed the query
    query_embedding = model.encode([query]).tolist()[0]

    # Query ChromaDB for the top 2 nearest documents
    resulted_docs = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )


    if 'documents' not in resulted_docs:
        return resulted_docs

    # Print the most similar documents
    return resulted_docs['documents'][0]
