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

    def query(self, question, top_k=6, min_score=0.7):
        """
        Performs a similarity search with score thresholding and returns relevant sources.
        
        Args:
            question: The user's question
            top_k: Number of chunks to retrieve
            min_score: Minimum relevance score (0-1) for a chunk to be included
            
        Returns:
            tuple: (filtered_docs, response_generator)
        """
        # Convert to vector embedding for similarity search
        embedding_function = self.vector_db._embedding_function
        query_embedding = embedding_function.embed_query(question)
        
        # Get documents with scores
        results = self.vector_db.similarity_search_with_score(
            question,
            k=top_k
        )
        
        # Filter by score threshold
        filtered_docs = [doc for doc, score in results if score >= min_score]
        
        if not filtered_docs:
            return [], iter(["I couldn't find relevant information in the document to answer this question."])
        
        # Generate and stream the response
        response = self.rag_chain.stream({
            "context": "\n\n".join([doc.page_content for doc in filtered_docs]),
            "question": question
        })
        
        return filtered_docs, response
