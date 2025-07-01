import streamlit as st
import os
import yaml
from src.rag_pipeline import RAGPipeline

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="PDF Q&A with Open-Source RAG", layout="wide")

st.title("📄 PDF Q&A with an Open-Source LLM")
st.write("Ask any question about your document. This app uses a local RAG pipeline with Ollama.")

# --- Helper Functions ---
@st.cache_resource
def load_pipeline():
    """Load the RAG pipeline, handling potential errors."""
    try:
        return RAGPipeline()
    except FileNotFoundError as e:
        st.error(f"**Error:** {e}")
        st.warning(f"Please make sure your PDF file is in the `data/` directory and named correctly in `config.yaml`.")
        return None
    except ValueError as e:
        st.error(f"**Error:** {e}")
        st.warning("This may be due to an issue with the PDF file itself.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {e}")
        return None

# --- Main App Logic ---
pipeline = load_pipeline()

if pipeline:
    st.sidebar.header("Controls")
    question = st.sidebar.text_input("Ask a question:", "")

    if st.sidebar.button("Get Answer", use_container_width=True) and question:
        with st.spinner("Finding an answer..."):
            try:
                answer = pipeline.query(question)
                st.success("Answer")
                st.markdown(answer)
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")
                st.warning("Please ensure the Ollama server is running and the model is available.")
    else:
        st.info("Enter a question in the sidebar and click 'Get Answer'.")
else:
    st.subheader("Pipeline failed to load.")
    st.write("Please resolve the errors displayed above to proceed.")
    if st.button("Retry Loading Pipeline"):
        st.cache_resource.clear()
        st.rerun()
