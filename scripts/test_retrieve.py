import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from  config import Config
from pubmed_retrieval import PubMedRetriever

retriever = PubMedRetriever(Config.EMAIL_FOR_PUBMED)
articles = retriever.search_and_retrieve("Parkinson's Disease", max_results=500)

with open('./data/N_abstracts.txt' , 'w', encoding="utf-8") as file_output:
    for article in articles:
        file_output.write(article['abstract'])
        file_output.write('\n')
    file_output.close()
print (len(articles))