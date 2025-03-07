import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.urls = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectors = None
        
    def add_document(self, text, url):
        self.documents.append(text)
        self.urls.append(url)
        self.vectors = self.vectorizer.fit_transform(self.documents)
        
    def search(self, query, top_k=3):
        if not self.documents:
            return []
            
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i in indices:
            if similarities[i] > 0.0:
                results.append({
                    'text': self.documents[i],
                    'url': self.urls[i],
                    'score': float(similarities[i])
                })
        
        return results

class WebScraper:
    def __init__(self, base_url, max_pages=10):
        self.base_url = base_url
        self.visited_urls = set()
        self.max_pages = max_pages
        parsed_url = urlparse(base_url)
        self.domain = parsed_url.netloc
        
    def is_valid_url(self, url):
        parsed = urlparse(url)
        return (parsed.netloc == self.domain or not parsed.netloc) and \
               parsed.scheme in ('http', 'https', '')
    
    def extract_text(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.decompose()
            
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_links(self, html, base_url):
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            url = link['href']
            absolute_url = urljoin(base_url, url)
            if self.is_valid_url(absolute_url):
                links.append(absolute_url)
                
        return links
    
    def scrape(self, url=None):
        if url is None:
            url = self.base_url
            
        if len(self.visited_urls) >= self.max_pages:
            return []
            
        if url in self.visited_urls:
            return []
            
        try:
            logger.info(f"Scraping {url}")
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }, timeout=10)
            response.raise_for_status()
            
            self.visited_urls.add(url)
            html = response.text
            
            text = self.extract_text(html)
            
            links = self.extract_links(html, url)
            
            results = [{'url': url, 'text': text}]
            
            for link in links:
                if len(self.visited_urls) < self.max_pages:
                    time.sleep(1)
                    results.extend(self.scrape(link))
                else:
                    break
                    
            return results
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []

class TextChunker:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text, url):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'url': url
                    })
                
                overlap_sentences = []
                overlap_length = 0
                
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_length = overlap_length + sentence_length
        
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'url': url
            })
            
        return chunks

class RAGPipeline:
    def __init__(self, api_key, model="qwen/qwq-32b:free", base_url="https://openrouter.ai/api/v1"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
        self.vector_db = SimpleVectorDB()
        
    def index_documents(self, documents):
        for doc in documents:
            self.vector_db.add_document(doc['text'], doc['url'])
        
        logger.info(f"Indexed {len(documents)} documents in the vector database")
        
    def retrieve(self, query, top_k=3):
        return self.vector_db.search(query, top_k=top_k)
        
    def generate(self, query, retrieved_docs):
        if not retrieved_docs:
            prompt = f"Please answer the following question: {query}"
        else:
            context = "\n\n".join([f"[Source: {doc['url']}]\n{doc['text']}" for doc in retrieved_docs])
            
            prompt = f"""Use the following information to answer the question. If the information provided doesn't contain the answer, say so.

INFORMATION:
{context}

QUESTION: {query}

ANSWER:"""
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://custom-rag-agent.com",
                    "X-Title": "Custom RAG Agent",
                },
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return {
                "answer": completion.choices[0].message.content,
                "sources": [doc['url'] for doc in retrieved_docs]
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": "Sorry, I encountered an error while generating a response.",
                "sources": []
            }

def main(base_url, api_key, query=None, max_pages=5):
    logger.info(f"Starting to scrape {base_url}")
    scraper = WebScraper(base_url, max_pages=max_pages)
    scraped_data = scraper.scrape()
    
    if not scraped_data:
        return "No data could be scraped from the provided URL."
    
    logger.info(f"Scraped {len(scraped_data)} pages from {base_url}")
    
    chunker = TextChunker()
    chunks = []
    for data in scraped_data:
        chunks.extend(chunker.chunk_text(data['text'], data['url']))
    
    logger.info(f"Created {len(chunks)} text chunks for indexing")
    
    rag_pipeline = RAGPipeline(api_key)
    
    rag_pipeline.index_documents(chunks)
    
    if query:
        retrieved_docs = rag_pipeline.retrieve(query)
        
        response = rag_pipeline.generate(query, retrieved_docs)
        
        return response
    else:
        return {
            "status": "success",
            "message": f"Successfully scraped and indexed {len(chunks)} text chunks from {len(scraped_data)} pages.",
            "ready_for_queries": True
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Scraping and RAG Pipeline")
    parser.add_argument("--url", required=True, help="Base URL to scrape")
    parser.add_argument("--api_key", required=True, help="OpenRouter API Key")
    parser.add_argument("--query", help="Query to run against the RAG pipeline")
    parser.add_argument("--max_pages", type=int, default=5, help="Maximum number of pages to scrape")
    
    args = parser.parse_args()
    
    result = main(args.url, args.api_key, args.query, args.max_pages)
    print(result)
