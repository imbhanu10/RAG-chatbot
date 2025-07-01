# RAG Pipeline for PDF Document Analysis (with Open-Source LLM)

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions based on a PDF document, using a locally-run open-source LLM via Ollama.

## Project Structure

```
project-root/
│
├── /data
│   ├── your_document.pdf      # Place your PDF here
│   ├── /chunks               # Stores text chunks from the PDF
│   └── /vectordb             # Stores the FAISS/Chroma index
├── /notebooks
│   ├── 01_pdf_analysis.ipynb      # EDA on the PDF
│   ├── 02_chunking_strategy.ipynb # Chunk size experiments
│   └── 03_embedding_eval.ipynb    # Embedding quality tests
├── /src
│   ├── __init__.py
│   ├── pdf_processor.py     # PDF parsing & chunking
│   ├── retriever.py         # Vector search
│   ├── generator.py         # LLM response generation (Ollama)
│   └── rag_pipeline.py      # Main pipeline
├── app.py                   # Streamlit interface
├── config.yaml              # Settings (chunk size, model names)
├── requirements.txt
├── .gitignore
├── README.md
└── report.md                # Placeholder for a report
```

## How to Run

1.  **Install and Run Ollama**:
    - Download and install [Ollama](https://ollama.com/).
    - Pull the model specified in `config.yaml` (default is `mistral`):
      ```bash
      ollama pull mistral
      ```
    - Make sure the Ollama application is running in the background.

2.  **Place your PDF:** Add your PDF file to the `data/` directory.

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
