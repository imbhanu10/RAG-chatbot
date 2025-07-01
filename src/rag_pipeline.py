import os
import yaml
from .pdf_processor import process_pdf
from .retriever import create_vector_db, load_vector_db
from .generator import get_rag_chain

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class RAGPipeline:
    def __init__(self, config):
        self.config = config
        db_dir = config['paths']['db_dir']
        if not os.path.exists(db_dir) or not os.listdir(db_dir):
            print("Vector database not found. Building from PDF...")
            documents = process_pdf(config['paths']['pdf_file'])
            if not documents:
                raise ValueError("No text could be extracted from the PDF. The pipeline cannot be initialized.")
            self.vector_db = create_vector_db(documents)
        else:
            self.vector_db = load_vector_db()
        
        self.rag_chain = get_rag_chain()
        print(f"RAG Pipeline initialized with {self.vector_db._collection.count()} document chunks")

    def query(self, question):
        """Performs a similarity search and streams the response, returning sources."""
        print(f"Searching for relevant context for: '{question}'")
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        print("Generating answer stream...")
        # The stream is a generator of tokens
        stream = self.rag_chain.stream({"context": context, "question": question})
        
        # Return the source documents and the stream generator
        return docs, stream
