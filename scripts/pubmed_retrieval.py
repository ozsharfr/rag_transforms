import time
import requests
import xml.etree.ElementTree as ET
from Bio import Entrez
import pandas as pd
from typing import List, Dict, Optional
import json

from config import Config

# Set your email for NCBI (required for API usage)
Entrez.email = "your.email@example.com"  # Replace with your actual email

class PubMedRetriever:
    def __init__(self, email: str, retmax: int = 100):
        """
        Initialize PubMed retriever
        
        Args:
            email: Your email address (required by NCBI)
            retmax: Maximum number of results to retrieve per search
        """
        Entrez.email = email
        self.retmax = retmax
        
    def search_pubmed(self, keyword: str, retmax: Optional[int] = None) -> List[str]:
        """
        Search PubMed for articles matching keyword
        
        Args:
            keyword: Search term or query
            retmax: Override default max results
            
        Returns:
            List of PubMed IDs
        """
        if retmax is None:
            retmax = self.retmax
            
        try:
            # Search PubMed
            search_handle = Entrez.esearch(
                db="pubmed",
                term=keyword,
                retmax=retmax,
                sort="relevance" #""
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            return search_results["IdList"]
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[Dict]:
        """
        Fetch abstracts for given PubMed IDs
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of dictionaries containing article information
        """
        if not pmids:
            return []
            
        articles = []
        
        try:
            # Fetch article details
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=pmids,
                rettype="medline",
                retmode="xml"
            )
            records = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            # Parse each article
            for record in records['PubmedArticle']:
                article_info = self._parse_article(record)
                if article_info:
                    articles.append(article_info)
                    
        except Exception as e:
            print(f"Error fetching abstracts: {e}")
            
        return articles
    
    def _parse_article(self, record) -> Optional[Dict]:
        """Parse individual article record"""
        try:
            article = record['MedlineCitation']['Article']
            
            # Extract basic information
            pmid = record['MedlineCitation']['PMID']
            title = article.get('ArticleTitle', 'No title available')
            
            # Extract abstract
            abstract_sections = article.get('Abstract', {}).get('AbstractText', [])
            if isinstance(abstract_sections, list):
                abstract = ' '.join([str(section) for section in abstract_sections])
            else:
                abstract = str(abstract_sections) if abstract_sections else 'No abstract available'
            
            # Extract authors
            authors = []
            if 'AuthorList' in article:
                for author in article['AuthorList']:
                    if 'LastName' in author and 'ForeName' in author:
                        authors.append(f"{author['ForeName']} {author['LastName']}")
            
            # Extract journal and publication date
            journal = article.get('Journal', {}).get('Title', 'Unknown journal')
            
            pub_date = record['MedlineCitation'].get('DateCompleted', {})
            if pub_date:
                year = pub_date.get('Year', 'Unknown')
                month = pub_date.get('Month', 'Unknown')
                day = pub_date.get('Day', 'Unknown')
                publication_date = f"{year}-{month}-{day}"
            else:
                publication_date = 'Unknown date'
            
            return {
                'pmid': str(pmid),
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'publication_date': publication_date,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None
    
    def search_and_retrieve(self, keyword: str, max_results: int = 50) -> List[Dict]:
        """
        Complete workflow: search and retrieve abstracts
        
        Args:
            keyword: Search term
            max_results: Maximum number of results
            
        Returns:
            List of article dictionaries with abstracts
        """
        print(f"Searching PubMed for: '{keyword}'")
        pmids = self.search_pubmed(keyword, retmax=max_results)
        print(f"Found {len(pmids)} articles")
        
        if pmids:
            print("Fetching abstracts...")
            articles = self.fetch_abstracts(pmids)
            print(f"Retrieved {len(articles)} abstracts")
            return articles
        else:
            print("No articles found")
            return []
    
    def save_to_csv(self, articles: List[Dict], filename: str):
        """Save articles to CSV file"""
        if articles:
            df = pd.DataFrame(articles)
            df.to_csv(filename, index=False)
            print(f"Saved {len(articles)} articles to {filename}")
        else:
            print("No articles to save")
    
    def save_to_json(self, articles: List[Dict], filename: str):
        """Save articles to JSON file"""
        if articles:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(articles)} articles to {filename}")
        else:
            print("No articles to save")

# Alternative method using direct API calls (no external dependencies)
class PubMedAPIRetriever:
    def __init__(self, email: str):
        self.email = email
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def search_and_retrieve(self, keyword: str, max_results: int = 50) -> List[Dict]:
        """Search and retrieve using direct API calls"""
        
        # Step 1: Search for PMIDs
        search_url = f"{self.base_url}esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': keyword,
            'retmax': max_results,
            'retmode': 'json'
        }
        
        try:
            response = requests.get(search_url, params=search_params)
            response.raise_for_status()
            search_data = response.json()
            pmids = search_data['esearchresult']['idlist']
            
            if not pmids:
                print("No articles found")
                return []
            
            print(f"Found {len(pmids)} articles")
            
            # Step 2: Fetch abstracts
            fetch_url = f"{self.base_url}efetch.fcgi"
            pmid_string = ','.join(pmids)
            
            fetch_params = {
                'db': 'pubmed',
                'id': pmid_string,
                'rettype': 'abstract',
                'retmode': 'xml'
            }
            
            response = requests.get(fetch_url, params=fetch_params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            articles = []
            
            for article in root.findall('.//PubmedArticle'):
                article_info = self._parse_xml_article(article)
                if article_info:
                    articles.append(article_info)
            
            return articles
            
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def _parse_xml_article(self, article_elem) -> Optional[Dict]:
        """Parse XML article element"""
        try:
            # Extract PMID
            pmid_elem = article_elem.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else 'Unknown'
            
            # Extract title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else 'No title'
            
            # Extract abstract
            abstract_elems = article_elem.findall('.//AbstractText')
            abstract_parts = [elem.text for elem in abstract_elems if elem.text]
            abstract = ' '.join(abstract_parts) if abstract_parts else 'No abstract available'
            
            # Extract authors
            author_elems = article_elem.findall('.//Author')
            authors = []
            for author in author_elems:
                fname = author.find('ForeName')
                lname = author.find('LastName')
                if fname is not None and lname is not None:
                    authors.append(f"{fname.text} {lname.text}")
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else 'Unknown journal'
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            
        except Exception as e:
            print(f"Error parsing article: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # Method 1: Using Biopython (recommended)
    print("=== Using Biopython Method ===")
    retriever = PubMedRetriever("your.email@example.com")  # Replace with your email
    
    # Search for articles about machine learning in healthcare
    keyword = "machine learning healthcare"
    articles = retriever.search_and_retrieve(keyword, max_results=10)
    
    # Display results
    for i, article in enumerate(articles[:3], 1):  # Show first 3 articles
        print(f"\n--- Article {i} ---")
        print(f"PMID: {article['pmid']}")
        print(f"Title: {article['title']}")
        print(f"Authors: {', '.join(article['authors'][:3])}{'...' if len(article['authors']) > 3 else ''}")
        print(f"Journal: {article['journal']}")
        print(f"Abstract: {article['abstract'][:200]}...")
        print(f"URL: {article['url']}")
    
    # Save results
    if articles:
        retriever.save_to_csv(articles, "pubmed_results.csv")
        retriever.save_to_json(articles, "pubmed_results.json")
    
    print(f"\n=== Alternative API Method ===")
    # Method 2: Using direct API calls
    api_retriever = PubMedAPIRetriever("your.email@example.com")
    api_articles = api_retriever.search_and_retrieve("COVID-19 vaccine", max_results=5)
    
    print(f"Retrieved {len(api_articles)} articles using API method")
