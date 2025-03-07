# Web Scraping RAG Agent

A powerful web scraping and question-answering system that combines web crawling with RAG (Retrieval-Augmented Generation) capabilities to provide intelligent answers based on website content.

## Features

- **Intelligent Web Scraping**: 
  - Scrapes text content from websites while respecting domain boundaries
  - Handles navigation through internal links
  - Automatically removes non-content elements (scripts, styles, headers, footers)
  - Rate-limited scraping to be respectful to websites
  - Configurable maximum page limit

- **Advanced Text Processing**:
  - Splits content into manageable chunks with configurable overlap
  - Uses NLTK for intelligent sentence tokenization
  - Implements TF-IDF vectorization for semantic search
  - Maintains source URLs for attribution

- **RAG Pipeline**:
  - Vector database for efficient document retrieval
  - Integration with OpenRouter API for LLM-powered responses
  - Context-aware answer generation
  - Source attribution for transparency

## Requirements

```
beautifulsoup4
nltk
numpy
openai
requests
scikit-learn
```

## Installation

1. Clone this repository

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK data:
   ```bash
   # Method 1: Using Python command
   python -c "import nltk; nltk.download('punkt')"

   # Method 2: Using Python interactive console
   python
   >>> import nltk
   >>> nltk.download('punkt')
   ```

4. Make sure you have an OpenRouter API key

## Usage

### Basic Usage

```bash
python rag_agent.py --url "https://example.com" --api_key "your-openrouter-api-key" --max_pages 5
```

### With a Query

```bash
python rag_agent.py --url "https://example.com" --api_key "your-openrouter-api-key" --query "What are the main products?" --max_pages 5
```

### Arguments

- `--url`: The base URL to start scraping from (required)
- `--api_key`: Your OpenRouter API key (required)
- `--query`: Question to ask about the scraped content (optional)
- `--max_pages`: Maximum number of pages to scrape (default: 5)

## Website Compatibility

This scraper works best with:
- Documentation websites
- Blog platforms
- Company websites
- Knowledge bases
- Educational resources
- Product information pages

The scraper respects:
- Same-domain boundaries (won't crawl external links)
- Common web scraping etiquette (rate limiting)
- Website structure (navigation through internal links)

## Limitations

- Only scrapes text content (no images or multimedia)
- Stays within the same domain
- May be blocked by websites with strict anti-scraping measures
- Requires JavaScript-rendered content to be accessible without JS execution

## Best Practices

1. Always check a website's robots.txt before scraping
2. Use reasonable delays between requests (built-in 1-second delay)
3. Don't set max_pages too high to avoid overwhelming servers
4. Ensure you have permission to scrape the target website

## License

MIT License

## Disclaimer

This tool should be used responsibly and in accordance with the target website's terms of service and robots.txt directives. Users are responsible for ensuring they have permission to scrape target websites. 