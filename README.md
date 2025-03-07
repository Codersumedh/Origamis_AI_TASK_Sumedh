# Text Scraping & Retrieval-Augmented Generation (RAG)

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
   ```
   python -c "import nltk; nltk.download('punkt')"
   python -c "import nltk; nltk.download('punkt_tab')"

   ```


## Usage


```bash
streamlit run app.py
```


Add following API KEY in frotend
```bash
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
```
Enter website link and start scrapping 
Use Query system section to ask Query on website data
