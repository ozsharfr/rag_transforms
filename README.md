# Medical Document Chat System

A RAG (Retrieval Augmented Generation) system for querying medical documents using advanced NLP techniques. The system processes medical papers/abstracts and allows users to ask questions through a chat interface.

## Features

- **Document Processing**: Automatically chunks and embeds medical documents
- **Smart Retrieval**: Uses sentence transformers and ChromaDB for efficient document retrieval
- **Medical Focus**: Optimized for biomedical content with PubMedBERT embeddings
- **Chat Interface**: Streamlit-based web interface for easy interaction
- **Caching**: Intelligent caching of embeddings for faster subsequent queries
- **Flexible Backend**: Supports both Ollama and Groq LLM providers

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   FILE_PATH=path/to/your/medical/documents.txt
   OLLAMA_HOST=http://localhost:11434  # if using Ollama
   ```

3. **Run the application**:
   ```bash
   streamlit run app_streamlit.py
   ```

4. **Start chatting**:
   - Upload your medical documents to the specified file path
   - Ask questions like "What are the possible Parkinson treatments?"
   - View processing logs in the expandable section

## Configuration

Key settings in `config.py`:
- `CHUNK_SIZE`: Document chunk size (default: 600)
- `RETRIEVE_TOP_K`: Number of documents to retrieve (default: 5)
- `BOOL_CHROMADB`: Use ChromaDB vs in-memory embeddings
- `TRANSFORMER_MODEL`: Embedding model (default: PubMedBERT)

## Architecture

```
User Query → Document Chunking → Embedding → Retrieval → LLM Response
                                     ↓
                            ChromaDB/In-Memory Storage
```

## File Structure

- `app_streamlit.py` - Streamlit web interface
- `main.py` - Core RAG pipeline
- `config.py` - Configuration settings
- `transformers_embed.py` - Text embedding utilities
- `test_chroma.py` - ChromaDB integration
- `text_split.py` - Document chunking
- `prompts_formatted.py` - LLM prompt templates

## Requirements

- Python 3.8+
- GROQ API key (recommended) or Ollama setup
- Medical documents in text format

## Notes

- First run will be slower as it processes and embeds documents
- Subsequent queries use cached embeddings for faster response
- Optimized for medical/scientific content but works with general documents