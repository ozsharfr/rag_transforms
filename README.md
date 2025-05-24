

## LLM-Powered RAG Document Analyzer

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **Ollama**, and **custom scoring functions** to extract and score relevant answers from large documents — particularly designed for tasks like answering questions from medical or scientific texts.
It iterates over few chunk size possibilities , though this may not be crucial

---

##  Features

* **Document preprocessing**: Filters noise like author lines and merge conflicts.
* **Chunking**: Splits documents into overlapping chunks of varying sizes.
* **Prompting**: Uses initial prompting and refined RAG prompts.
* **Similarity search**: Retrieves top relevant chunks using a sentence similarity model.
* **LLM scoring**: Ranks retrieved chunks using an LLM.
* **Logging**: Tracks all steps with timestamped logs.

---

## 🧰 Technologies Used

* **LangChain** with `OllamaLLM` - model can be modified as requested
* **Transformers for sentence embedding**
* **Custom utilities** for chunking, filtering, logging
* **Ollama** for local LLM inference
* **Python 3.10+**

---

##  Project Structure

```
.
├── app.py                     # Wrap in app , if a docker is needed
├── main.py                    # Main pipeline script
├── config.py                  # Configuration variables (paths, model names, etc.)
├── transformers_embed.py      # Similarity search with sentence embeddings
├── prompts_formatted.py       # Prompt templates and LLM interaction logic
├── result_score.py            # Scoring functions using LLM
├── text_split.py              # Document chunking logic
├── utils/
│   ├── doc_parser.py          # Filters noise (author/conflict lines)
│   ├── file_reader.py         # Reads documents from disk
│   └── logger.py              # Logging setup
```

---

## 🧪 How It Works

1. **Read and clean the document** from a given file path.
2. **Preprocess the text** by removing irrelevant lines.
3. **Split into overlapping chunks** of varying sizes (400, 600, 800 tokens).
4. For each chunk size:

   * Query the LLM using an initial prompt.
   * Retrieve top relevant chunks using semantic similarity.
   * Score each chunk using the LLM.
   * Filter out low-relevance results.
   * Construct a final RAG prompt and get the final answer.
   * Log scores, answers, and timing.

---

## ⚙️ Configuration

All runtime parameters are stored in `config.py`, including:

* `FILE_PATH`: Path to the document
* `MODEL_NAME`: Name of the Ollama model (e.g., `llama3`, `mistral`)
* `OLLAMA_HOST`: URL to the local Ollama server
* `MIN_RELEVANCE_SCORE`: Threshold for filtering out low-quality chunks
* `LOG_FILE`: File path for logs
* `LOG_LEVEL`: Logging verbosity (e.g., `INFO`, `DEBUG`)

---

## 🧑‍💻 Running the Script

1. Start your Ollama server locally:

   ```bash
   ollama run llama3
   ```

2. Run the pipeline:

   ```bash
   python main.py
   ```

3. View results in your configured log file.

---

##  Example Use Case

Query:

> *"What are the possible Parkinson treatments?"*

The pipeline will return:

* Relevant text chunks
* LLM-generated answer based on retrieved context
* Scoring to rank result quality

---

##  Requirements

* Python 3.10+
* Install dependencies:

```bash
pip install -r requirements.txt
```

(You may need to add this file manually based on your project imports.)

---

##  Notes

* For local use only (with Ollama) — no external API calls.
* Designed for experimentation and fast iteration on document-based QA tasks.

---

Would you like me to generate a `requirements.txt` file as well based on your imports?
