import yaml
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def get_rag_chain():
    """Initializes and returns a RAG chain for response generation using Ollama."""
    template = """
    You are an expert assistant. Your task is to answer the user's question based ONLY on the following context.
    Synthesize the information from the context to provide a clear, concise, and conversational answer.
    Do NOT simply copy and paste sentences from the context. Rephrase the answer in your own words.
    If the context does not contain the answer, you must state that you cannot answer based on the provided document.

    Context:
    ---
    {context}
    ---

    Question: {question}

    Answer:
    """
    
    prompt = PromptTemplate.from_template(template)
    
    llm = Ollama(model=config['llm']['model_name'])
    
    rag_chain = (
        {"context": (lambda x: x['context']), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
