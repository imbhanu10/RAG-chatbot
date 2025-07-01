import yaml
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def get_rag_chain():
    """Initializes and returns a RAG chain for response generation using Ollama."""
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions STRICTLY based on the provided context. Follow these rules:
        1. Answer ONLY using the information from the provided context
        2. If the context doesn't contain the answer, say 'I couldn't find this information in the document.'
        3. Never make up or assume information that's not in the context
        4. Keep your answer concise and to the point
        5. If the question is not related to the document, say 'This question is not covered in the document.'
        
        Context: {context}
        Question: {question}
        
        Answer:"""),
        ("human", "{question}")
    ])
    prompt = PromptTemplate.from_template(template)
    
    llm = Ollama(model=config['llm']['model_name'])
    
    rag_chain = (
        {"context": (lambda x: x['context']), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
