import streamlit as st
import pandas as pd
import time
import os
import sys
from rag_agent import WebScraper, TextChunker, RAGPipeline, SimpleVectorDB


st.set_page_config(
    page_title="Knowledge Navigator - Web RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .data-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
        border-left: 3px solid #4361ee;
    }
    .response-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        flex: 1;
        margin: 0 10px;
        text-align: center;
    }
    .title-area {
        text-align: center;
        margin-bottom: 30px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #888;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'scraping_complete' not in st.session_state:
    st.session_state.scraping_complete = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

st.markdown("<div class='title-area'><h1>🔍 Knowledge Navigator</h1><h3>Web Scraping & Retrieval-Augmented Generation System</h3></div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("OpenRouter API Key", type="password", 
                           help="Enter your OpenRouter API key here")
    
    model_options = {
        "Qwen 32B": "qwen/qwq-32b:free",
    }
    
    selected_model = st.selectbox("Select LLM Model", 
                                  options=list(model_options.keys()),
                                  help="Choose which language model to use for generating responses")
    model = model_options[selected_model]
    
    st.divider()
    
    st.header("🔄 Workflow")
    
    st.info("1. Enter a website URL to scrape\n2. Configure scraping parameters\n3. Start scraping\n4. Ask questions about the content")
    
    st.divider()
    
    if st.session_state.query_history:
        st.header("📝 Query History")
        for idx, (q, _) in enumerate(st.session_state.query_history[-5:]):
            st.caption(f"{idx+1}. {q}")
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    Knowledge Navigator is a RAG system that:
    - Scrapes web content
    - Indexes it in a vector database
    - Retrieves relevant content for queries
    - Generates accurate responses with sources
    """)

tab1, tab2 = st.tabs(["🕸️ Web Scraping", "❓ Query System"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        website_url = st.text_input("Website URL to Scrape", 
                                  placeholder="https://example.com",
                                  help="Enter the full URL including https://")
    
    with col2:
        max_pages = st.number_input("Maximum Pages to Scrape", 
                                  min_value=1, max_value=50, value=5,
                                  help="Higher values will take longer but gather more information")
    
    with st.expander("Advanced Scraping Options"):
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("Text Chunk Size", min_value=100, max_value=2000, value=512)
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=50)
            
    if st.button("🚀 Start Scraping", use_container_width=True, type="primary", 
                disabled=not website_url or not api_key):
        with st.spinner("Scraping website... This may take a few minutes depending on site size."):
            try:
                st.session_state.scraping_complete = False
                st.session_state.scraped_data = None
                st.session_state.chunks = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Initializing scraper...")
                scraper = WebScraper(website_url, max_pages=max_pages)
                
                status_text.text("Scraping pages...")
                scraped_data = scraper.scrape()
                progress_bar.progress(40)
                
                if not scraped_data:
                    st.error("No data could be scraped from the provided URL.")
                else:
                    status_text.text("Processing and chunking text...")
                    chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
                    chunks = []
                    for data in scraped_data:
                        chunks.extend(chunker.chunk_text(data['text'], data['url']))
                    
                    progress_bar.progress(70)
                    
                    status_text.text("Setting up RAG pipeline and indexing documents...")
                    rag_pipeline = RAGPipeline(api_key, model=model)
                    
                    rag_pipeline.index_documents(chunks)
                    progress_bar.progress(100)
                    
                    st.session_state.scraped_data = scraped_data
                    st.session_state.chunks = chunks
                    st.session_state.rag_pipeline = rag_pipeline
                    st.session_state.scraping_complete = True
                    
                    status_text.empty()
                    
                    st.success(f"Successfully scraped and indexed {len(chunks)} text chunks from {len(scraped_data)} pages.")
                    
            except Exception as e:
                st.error(f"An error occurred during scraping: {str(e)}")
    
    if st.session_state.scraping_complete and st.session_state.scraped_data:
        st.divider()
        
        st.markdown("<h3>📊 Scraping Metrics</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Pages Scraped</h4>
                <h2>{len(st.session_state.scraped_data)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Text Chunks</h4>
                <h2>{len(st.session_state.chunks)}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            avg_chunk_length = sum(len(chunk['text']) for chunk in st.session_state.chunks) / max(1, len(st.session_state.chunks))
            st.markdown(f"""
            <div class='metric-card'>
                <h4>Avg. Chunk Size</h4>
                <h2>{int(avg_chunk_length)} chars</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📑 View Scraped Pages"):
            pages_df = pd.DataFrame([
                {"URL": data['url'], 
                 "Content Length": len(data['text']),
                 "Preview": data['text'][:100] + "..." if len(data['text']) > 100 else data['text']}
                for data in st.session_state.scraped_data
            ])
            
            st.dataframe(pages_df, use_container_width=True)
        
        with st.expander("🧩 View Text Chunks"):
            chunks_df = pd.DataFrame([
                {"Source URL": chunk['url'], 
                 "Length": len(chunk['text']),
                 "Content": chunk['text'][:100] + "..." if len(chunk['text']) > 100 else chunk['text']}
                for chunk in st.session_state.chunks
            ])
            
            st.dataframe(chunks_df, use_container_width=True)

with tab2:
    if not st.session_state.scraping_complete:
        st.warning("Please scrape a website first (in the Web Scraping tab) before asking questions.")
    else:
        st.markdown("<h3>🔎 Ask About the Scraped Content</h3>", unsafe_allow_html=True)
        
        query = st.text_input("Enter your question", 
                            placeholder="What does this website say about...?",
                            help="Ask a question about the content you've scraped")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            top_k = st.number_input("Number of sources to use", min_value=1, max_value=10, value=3)
        
        if st.button("🔍 Search", use_container_width=True, type="primary", disabled=not query):
            with st.spinner("Retrieving relevant information and generating response..."):
                try:
                    retrieved_docs = st.session_state.rag_pipeline.retrieve(query, top_k=top_k)
                    
                    start_time = time.time()
                    response = st.session_state.rag_pipeline.generate(query, retrieved_docs)
                    generation_time = time.time() - start_time
                    
                    st.session_state.query_history.append((query, response))
                    
                    st.markdown("<div class='response-container'>", unsafe_allow_html=True)
                    st.markdown("### 📝 Answer")
                    st.write(response["answer"])
                    
                    if response["sources"]:
                        st.markdown("### 📚 Sources")
                        for idx, source in enumerate(response["sources"]):
                            st.markdown(f"""
                            <div class='source-box'>
                                <strong>Source {idx+1}:</strong> <a href="{source}" target="_blank">{source}</a>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.caption(f"Response generated in {generation_time:.2f} seconds using {selected_model}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        
        if st.session_state.query_history:
            with st.expander("📜 View Previous Queries and Responses"):
                for i, (past_query, past_response) in enumerate(reversed(st.session_state.query_history)):
                    st.markdown(f"**Q{i+1}: {past_query}**")
                    st.markdown(f"A: {past_response['answer']}")
                    st.divider()

st.markdown("<div class='footer'>Knowledge Navigator</div>", unsafe_allow_html=True)