import os
import yaml
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def process_pdf(file_path):
    """Reads a PDF and splits it into text chunks."""
    print(f"Processing {file_path}...")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text:
        print("Warning: No text extracted from PDF.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['pdf_processing']['chunk_size'],
        chunk_overlap=config['pdf_processing']['chunk_overlap'],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    chunk_dir = config['paths']['chunk_dir']
    os.makedirs(chunk_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(os.path.join(chunk_dir, f"chunk_{i}.txt"), "w", encoding='utf-8') as chunk_file:
            chunk_file.write(chunk)
            
    print(f"Created {len(chunks)} chunks.")
    return chunks
