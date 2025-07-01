import yaml
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def create_vector_db(chunks):
    """Creates a ChromaDB vector store from text chunks."""
    print("Creating vector database...")
    model_name = config['embedding']['model_name']
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    db_dir = config['paths']['db_dir']
    vector_db = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        persist_directory=db_dir
    )
    vector_db.persist()
    print("Vector database created.")
    return vector_db

def load_vector_db():
    """Loads an existing ChromaDB vector store."""
    print("Loading vector database...")
    model_name = config['embedding']['model_name']
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db_dir = config['paths']['db_dir']
    vector_db = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    print("Vector database loaded.")
    return vector_db
