import os
import yaml
from .pdf_processor import process_pdf
from .retriever import create_vector_db, load_vector_db
from .generator import get_rag_chain

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class RAGPipeline:
    def __init__(self):
        db_dir = config['paths']['db_dir']
        if not os.path.exists(db_dir) or not os.listdir(db_dir):
            print("Vector database not found. Building from PDF...")
            pdf_path = config['paths']['pdf_file']
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found at {pdf_path}. Please add it.")
            chunks = process_pdf(pdf_path)
            if not chunks:
                raise ValueError("No text could be extracted from the PDF. The pipeline cannot be initialized.")
            self.vector_db = create_vector_db(chunks)
        else:
            self.vector_db = load_vector_db()
        
        self.rag_chain = get_rag_chain()

    def query(self, question):
        """Performs a similarity search and generates a response."""
        print(f"Searching for relevant context for: '{question}'")
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        print("Generating answer...")
        response = self.rag_chain.invoke({"context": context, "question": question})
        return response
