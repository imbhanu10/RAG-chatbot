import streamlit as st
import yaml
from src.rag_pipeline import RAGPipeline

# --- CONFIGURATION & INITIALIZATION ---
st.set_page_config(
    page_title="RAG Chatbot with Streaming",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

@st.cache_resource
def load_pipeline():
    """Loads the RAG pipeline and returns it along with the number of indexed docs."""
    try:
        config = load_config()
        pipeline = RAGPipeline(config)
        # Get the number of documents from the vector store
        num_docs = pipeline.vector_db._collection.count()
        return pipeline, num_docs
    except Exception as e:
        st.error(f"Failed to load the RAG pipeline: {e}")
        st.stop()

config = load_config()
llm_model_name = config.get('llm', {}).get('model_name', 'N/A')
pipeline, num_indexed_docs = load_pipeline()

# --- SIDEBAR ---
with st.sidebar:
    st.header("Chatbot Configuration")
    st.info(f"**Model in Use:** `{llm_model_name}`")
    st.success(f"**Indexed Documents:** `{num_indexed_docs}` chunks")
    
    if st.button("Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("💬 RAG Chatbot with Streaming Responses")
st.markdown("Ask questions about your document. The chatbot will answer and show you the sources it used.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Get the stream and source documents from the pipeline
            source_docs, response_stream = pipeline.query(prompt)
            
            # Stream the response to the UI
            for chunk in response_stream:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌") # Add a blinking cursor
            message_placeholder.markdown(full_response)

            # Display the source documents with page numbers and context
            st.divider()
            st.subheader("Sources Used")
            for i, doc in enumerate(source_docs, 1):
                with st.expander(f"Source {i} (Page {doc.metadata.get('page', 'N/A')})"):
                    st.caption(f"**Page {doc.metadata.get('page', 'N/A')}**")
                    st.markdown(doc.page_content)
                    st.caption("---")

        except Exception as e:
            full_response = f"An error occurred: {e}"
            message_placeholder.error(full_response)

    # Add the final assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
