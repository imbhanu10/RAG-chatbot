import os
import yaml
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def process_pdf(file_path):
    """Reads a PDF and splits it into text chunks with page numbers."""
    print(f"Processing {file_path}...")
    reader = PdfReader(file_path)
    documents = []
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text:
            # Split page text into sentences for better chunking
            sentences = page_text.replace('\n', ' ').split('.')
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < config['pdf_processing']['chunk_size']:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        doc = Document(
                            page_content=current_chunk.strip(),
                            metadata={"page": page_num, "source": f"Page {page_num}"}
                        )
                        documents.append(doc)
                    current_chunk = sentence + ". "
            
            # Add the last chunk of the page
            if current_chunk.strip():
                doc = Document(
                    page_content=current_chunk.strip(),
                    metadata={"page": page_num, "source": f"Page {page_num}"}
                )
                documents.append(doc)
    
    chunk_dir = config['paths']['chunk_dir']
    os.makedirs(chunk_dir, exist_ok=True)
    for i, document in enumerate(documents):
        with open(os.path.join(chunk_dir, f"chunk_{i}.txt"), "w", encoding='utf-8') as chunk_file:
            chunk_file.write(document.page_content)
            
    print(f"Created {len(documents)} chunks.")
    return documents
