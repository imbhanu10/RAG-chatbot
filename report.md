# Project Report: RAG Pipeline Analysis


## 1. Description of Document Structure and Chunking Logic

The pipeline is designed to work with a single PDF document as its knowledge base.

- **Document Processing**: The system begins by loading the PDF located at `data/AI Training Document.pdf`. Text extraction is handled by the `PyPDF2` library, which reads all pages sequentially to obtain the raw text.

- **Chunking Strategy**: After extraction, the full text is segmented using LangChain's `RecursiveCharacterTextSplitter`, which intelligently splits content at semantic boundaries (e.g., paragraphs > sentences > words).

- **Configuration**:
  - `chunk_size: 1000`: Each chunk contains up to 1000 characters.
  - `chunk_overlap: 200`: Overlapping of 200 characters between consecutive chunks ensures continuity and reduces context fragmentation.

---

## 2. Explanation of Embedding Model and Vector DB Used

To enable semantic retrieval, each chunk is converted into an embedding and stored in a vector database.

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - Outputs 384-dimensional dense vectors.
  - Lightweight and CPU-friendly, suitable for small to mid-sized RAG tasks.
  - Captures semantic meaning well for English sentences.

- **Vector Database**: `ChromaDB`
  - Open-source, lightweight, and local vector database.
  - Used to perform similarity searches to retrieve relevant chunks at inference time.
  - Stores and indexes embeddings in `data/vectordb/`.

---

## 3. Prompt Format and Generation Logic

The system uses a Retrieval-Augmented Generation (RAG) architecture with an instruct-tuned open-source LLM.

- **Retriever**: Fetches top 4 most relevant chunks from ChromaDB based on cosine similarity to the query embedding.

- **Prompt Template**:

    ```
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Helpful Answer:
    ```

- This prompt structure encourages grounded answers and explicitly instructs the model not to hallucinate.

- **LLM Used**: `mistral-7b-instruct` (loaded via Hugging Face Transformers or Ollama)

---

## 4. Example Queries with Responses

Below are examples representing both successful and failure cases.

### ✅ Successful Retrieval
**Query**: *"What rights do I grant by providing my data?"*  
**Response**: *"When you provide content on eBay's platform, you grant a worldwide, perpetual, irrevocable, royalty-free, sublicensable (through multiple tiers) right to exercise any and all Intellectual Property Rights you have in that content. This right is connected with their provision, expansion, and promotion of their services, including the development of new offerings as part of their services, in any media known now or developed in the future. Additionally, you waive your right to enforce your Intellectual Property Rights, such as moral rights, against eBay, their assignees, sublicensees, and their assignees, in connection with their use of that content for these purposes. This means that they have the freedom to use and modify your content without requiring your consent or payment, as long as it's compliant with their policies and user agreement."*


### ✅ Successful Retrieval
**Query**: *"Is the training data stored in Europe?"*  
**Response**: *"Based on the provided context, it doesn't appear that the training data is explicitly mentioned or stored in Europe. The document discusses various aspects of eBay's international policies, including international buying and selling, translation services, and payment information. However, there's no information suggesting that eBay stores its training data in Europe. If you require more specific information about eBay's data storage practices, I would recommend checking the Privacy Policy section on their official website for details."*

### ✅ Successful Reasoning
**Query**: *"What is the penalty for violating the agreement?"*  
**Response**: *"The User Agreement does not specify a monetary penalty for violating it. However, there are sections that outline potential consequences such as indemnification and release of claims against the platform (eBay in this case). If a dispute arises with another user, you are required to release the platform from any claims, demands, and damages arising out of or in any way connected with such disputes. Furthermore, you will indemnify and hold the platform liable for any consequential damages or losses resulting directly or indirectly from the content you provide using their services or your use of or inability to use their services. It's essential to read the full User Agreement for a comprehensive understanding of its terms."*  
*Comment: This would be a hallucination if the document doesn't state a specific fine. This highlights how vague prompts or missing context can trigger fabricated answers.*

---

## 5. Notes on Hallucinations, Model Limitations, and Slow Responses



- **Model Limitations**:
  - The model is **fully dependent on the retriever**. If irrelevant or incomplete chunks are retrieved, the answer quality deteriorates.
  - Mistral-7B has limited internal knowledge and performs best when provided with direct, relevant context.

- **Response Latency**: Running locally without GPU can result in slower generation times, especially for larger queries. Streaming is implemented to improve UX by showing responses as they're generated.

---

