# RAG Chatbot with Document Analysis

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on a provided PDF document. This implementation uses:

- **Mistral 7B** (via Ollama) as the language model
- **ChromaDB** for vector storage and retrieval
- **Sentence Transformers** for document embeddings
- **Streamlit** for the web interface

## ✨ Features

- **Document Processing**: Automatically processes and chunks PDF documents with page-level tracking
- **Semantic Search**: Finds relevant document sections using vector similarity
- **Source Citation**: Shows exact page numbers and source text for all answers
- **Streaming Responses**: Real-time token streaming for a better user experience
- **Persistent Storage**: Saves processed documents for faster subsequent loads

## 🚀 Quick Start

### Prerequisites

1. Install [Ollama](https://ollama.ai/) and pull the Mistral model:
   ```bash
   ollama pull mistral
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. Place your PDF document in the `data/` directory (or update the path in `config.yaml`)
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to `http://localhost:8501`

## 🛠️ Configuration

Edit `config.yaml` to customize:

```yaml
pdf_processing:
  chunk_size: 750      # Adjust based on document complexity
  chunk_overlap: 200   # Overlap between chunks for better context

embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"

llm:
  model_name: "mistral"  # Using Mistral 7B via Ollama

paths:
  pdf_file: "data/your_document.pdf"
  chunk_dir: "data/chunks"
  db_dir: "data/vectordb"
```

## 📂 Project Structure

```
rag-project/
├── data/                  # Store PDFs and processed chunks
├── notebooks/             # Jupyter notebooks for analysis
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py   # PDF text extraction and chunking
│   ├── retriever.py       # Vector database management
│   ├── generator.py       # LLM response generation
│   └── rag_pipeline.py    # Main RAG pipeline
├── app.py                # Streamlit web interface
├── config.yaml           # Configuration settings
└── requirements.txt      # Python dependencies
```

## 🤖 How It Works

1. **Document Processing**:
   - Extracts text from PDF with page numbers
   - Splits content into overlapping chunks
   - Stores metadata including source page numbers

2. **Query Processing**:
   - Converts user questions into vector embeddings
   - Retrieves most relevant document sections
   - Generates answers using the LLM with proper citations

3. **Response Generation**:
   - Streams responses token by token
   - Includes source citations with page numbers
   - Preserves chat history during the session

## 📝 Notes

- First run will take longer as it processes the entire document
- The application maintains a local vector database for faster subsequent loads
- For large documents, consider increasing the chunk size in the config

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
