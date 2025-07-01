import yaml
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def create_vector_db(documents):
    """Creates and persists a vector database from documents with metadata."""
    print("Creating vector database...")
    embedding = HuggingFaceEmbeddings(model_name=config['embedding']['model_name'])
    
    # Extract texts and metadatas
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # Create and persist the vector store with metadata
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory=config['paths']['db_dir'],
        collection_metadata={"hnsw:space": "cosine"}  # Better for semantic search
    )
    
    print(f"Vector database created with {len(documents)} documents.")
    return vector_db

def load_vector_db():
    """Loads an existing ChromaDB vector store with metadata support."""
    print("Loading existing vector database...")
    embedding = HuggingFaceEmbeddings(model_name=config['embedding']['model_name'])
    
    vector_db = Chroma(
        persist_directory=config['paths']['db_dir'],
        embedding_function=embedding,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Verify the collection was loaded with metadata
    if vector_db._collection is not None:
        print(f"Vector database loaded with {vector_db._collection.count()} documents.")
    else:
        print("Warning: Vector database collection is empty or not properly loaded.")
        
    return vector_db
