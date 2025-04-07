# RAG System with Claude and Streamlit
# This application extracts text from PDFs, creates embeddings, and uses Claude to answer questions
# about naturalness based on the retrieved content.

import streamlit as st
import anthropic
import os
import tempfile
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import time

# Set page configuration
st.set_page_config(
    page_title="PDF RAG with Claude",
    page_icon="📚",
    layout="wide",
)

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Smaller, faster model for embeddings
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
MAX_CHUNKS_TO_USE = 5  # Maximum number of chunks to use for context

# Initialize the Claude client
@st.cache_resource
def get_claude_client():
    api_key = st.session_state.get("api_key", "")
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    return None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to chunk text
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and end - start >= chunk_size:
            # Find the last period or newline to make clean breaks
            last_period = text.rfind('.', start, end)
            last_newline = text.rfind('\n', start, end)
            if last_period != -1 and last_period > start + chunk_size // 2:
                end = last_period + 1
            elif last_newline != -1 and last_newline > start + chunk_size // 2:
                end = last_newline + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end < text_length else text_length
    
    return chunks

# Function to create embeddings
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def create_embeddings(chunks, model):
    return model.encode(chunks)

# Function to find relevant chunks
def get_relevant_chunks(query, chunks, embeddings, embedding_model, top_k=MAX_CHUNKS_TO_USE):
    query_embedding = embedding_model.encode([query])[0]
    
    # Calculate cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get indices of top-k most similar chunks
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [chunks[i] for i in top_indices], similarities[top_indices]

# Function to generate response from Claude
def generate_response(query, relevant_chunks, claude_client):
    # Create context from relevant chunks
    context = "\n\n".join(relevant_chunks)
    
    # Craft prompt for Claude
    prompt = f"""You are an assistant specialized in answering questions about naturalness based on the provided document.
Use only the information in the following text to answer the question. If the answer is not in the text, say "I don't have enough information from the document to answer this question."

DOCUMENT TEXT:
{context}

QUESTION:
{query}

Provide a detailed and accurate answer based only on the document content."""

    try:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Save and load RAG data
def save_rag_data(chunks, embeddings, file_name):
    data = {
        "chunks": chunks,
        "embeddings": embeddings,
        "file_name": file_name,
    }
    with open("rag_data.pkl", "wb") as f:
        pickle.dump(data, f)

def load_rag_data():
    if os.path.exists("rag_data.pkl"):
        with open("rag_data.pkl", "rb") as f:
            return pickle.load(f)
    return None

# Sidebar for configuration
with st.sidebar:
    st.title("Configuration")
    
    # API Key Input
    api_key_input = st.text_input(
        "Enter your Anthropic API key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Get your API key from https://console.anthropic.com/",
    )
    
    if api_key_input:
        st.session_state["api_key"] = api_key_input
    
    # PDF Upload
    uploaded_file = st.file_uploader("Upload a PDF about naturalness", type="pdf")
    
    if uploaded_file is not None:
        process_button = st.button("Process PDF")
        st.info("After uploading, click 'Process PDF' to extract text and create embeddings")

# Main app
st.title("RAG System with Claude for Naturalness")

# Initialize or load data
if "is_processed" not in st.session_state:
    rag_data = load_rag_data()
    if rag_data:
        st.session_state["chunks"] = rag_data["chunks"]
        st.session_state["embeddings"] = rag_data["embeddings"]
        st.session_state["file_name"] = rag_data["file_name"]
        st.session_state["is_processed"] = True
    else:
        st.session_state["is_processed"] = False

# Process PDF if requested
if uploaded_file is not None and "process_button" in locals() and process_button:
    with st.spinner("Processing PDF file... This may take a minute."):
        # Extract text from PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        text = extract_text_from_pdf(tmp_path)
        os.unlink(tmp_path)  # Delete temporary file
        
        # Chunk text
        chunks = chunk_text(text)
        
        # Create embeddings
        embedding_model = get_embedding_model()
        embeddings = create_embeddings(chunks, embedding_model)
        
        # Save to session state
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["is_processed"] = True
        
        # Save data for future sessions
        save_rag_data(chunks, embeddings, uploaded_file.name)
        
        st.success(f"Processed {len(chunks)} chunks from {uploaded_file.name}")
        st.experimental_rerun()

# Display file information if processed
if st.session_state.get("is_processed", False):
    st.info(f"Working with document: {st.session_state.get('file_name', 'Unknown')}")
    st.text(f"Document contains {len(st.session_state['chunks'])} chunks of text")

# Query input section
if st.session_state.get("is_processed", False):
    st.subheader("Ask a question about naturalness")
    query = st.text_input("Your question")
    
    if query:
        client = get_claude_client()
        
        if not client:
            st.error("Please enter a valid Anthropic API key in the sidebar")
        else:
            # Get relevant chunks
            with st.spinner("Finding relevant information..."):
                relevant_chunks, similarities = get_relevant_chunks(
                    query,
                    st.session_state["chunks"],
                    st.session_state["embeddings"],
                    get_embedding_model()
                )
            
            # Generate response
            with st.spinner("Generating response with Claude..."):
                response = generate_response(query, relevant_chunks, client)
            
            # Display response
            st.subheader("Response")
            st.write(response)
            
            # Show sources (optional)
            with st.expander("Sources (Relevant Text Chunks)"):
                for i, (chunk, similarity) in enumerate(zip(relevant_chunks, similarities)):
                    st.markdown(f"**Chunk {i+1}** (Relevance: {similarity:.2f})")
                    st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                    st.markdown("---")
else:
    if uploaded_file is None:
        st.info("Please upload a PDF document about naturalness in the sidebar")
    else:
        st.info("Click 'Process PDF' in the sidebar to start")

# Footer
st.markdown("---")
st.markdown(
    "This application uses Claude to answer questions about naturalness based on the provided PDF document."
)
