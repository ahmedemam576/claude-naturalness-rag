# Enhanced RAG System with OpenAI and Streamlit
# This application uses a predefined PDF about naturalness, creates embeddings,
# and uses OpenAI to answer questions based on the retrieved content.

import streamlit as st
import openai
import os
import tempfile
from pypdf import PdfReader
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import time
import base64
from pathlib import Path
import threading

# Set page configuration
st.set_page_config(
    page_title="Naturalness Q&A with AI",
    page_icon="🌿",
    layout="wide",
)

# Apply custom CSS for a more appealing interface
def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px 15px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #2E7D32;
    }
    .response-container {
        background-color: white;
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .source-container {
        background-color: #f1f3f4;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    .sidebar .stTextInput>div>div>input {
        background-color: white;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Smaller, faster model for embeddings
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
MAX_CHUNKS_TO_USE = 5  # Maximum number of chunks to use for context
PDF_PATH = "naturalness.pdf"  # Path to the predefined PDF

# Logo and header
def create_header():
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("https://img.icons8.com/color/96/000000/natural-food.png", width=80)
    
    with col2:
        st.title("Naturalness AI Assistant")
        st.subheader("Ask questions about naturalness and get AI-powered answers")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

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

# Function to generate response from OpenAI - using older API style for compatibility
# Function to generate response from OpenAI - compatible with OpenAI v1.0+
def generate_response(query, relevant_chunks):
    # Create context from relevant chunks
    context = "\n\n".join(relevant_chunks)
    
    # Craft prompt for OpenAI
    prompt = f"""You are an assistant specialized in answering questions about naturalness based on the provided document.
Use only the information in the following text to answer the question. If the answer is not in the text, say "I don't have enough information from the document to answer this question."

DOCUMENT TEXT:
{context}

QUESTION:
{query}

Provide a detailed and accurate answer based only on the document content."""

    try:
        # Import the OpenAI client from the new package
        from openai import OpenAI
        
        # Initialize the client with your API key
        client = OpenAI(api_key=st.session_state.get("api_key", ""))
        
        # Use the new API format
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided document content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        
        # Access the content in the new response format
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Save and load RAG data
def save_rag_data(chunks, embeddings):
    data = {
        "chunks": chunks,
        "embeddings": embeddings,
    }
    with open("rag_data.pkl", "wb") as f:
        pickle.dump(data, f)

def load_rag_data():
    if os.path.exists("rag_data.pkl"):
        with open("rag_data.pkl", "rb") as f:
            return pickle.load(f)
    return None

# Process the PDF and generate embeddings
def process_pdf():
    # Create a placeholder for the progress bar
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Check if PDF exists
        status_text.text("Checking for PDF file...")
        progress_bar.progress(10)
        time.sleep(0.5)  # Small delay for visual feedback
        
        if not os.path.exists(PDF_PATH):
            progress_placeholder.empty()
            status_text.empty()
            st.error(f"PDF file not found: {PDF_PATH}")
            return False
        
        # Step 2: Extract text from PDF
        status_text.text("Extracting text from PDF...")
        progress_bar.progress(20)
        text = extract_text_from_pdf(PDF_PATH)
        
        if not text:
            progress_placeholder.empty()
            status_text.empty()
            return False
        
        # Step 3: Chunking text
        status_text.text("Breaking text into meaningful segments...")
        progress_bar.progress(40)
        chunks = chunk_text(text)
        
        # Step 4: Loading embedding model
        status_text.text("Loading language understanding model...")
        progress_bar.progress(60)
        embedding_model = get_embedding_model()
        
        # Step 5: Creating embeddings
        status_text.text("Creating vector embeddings for text segments...")
        progress_bar.progress(75)
        embeddings = create_embeddings(chunks, embedding_model)
        
        # Step 6: Saving data
        status_text.text("Saving processed data...")
        progress_bar.progress(90)
        
        # Save to session state
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings
        st.session_state["is_processed"] = True
        
        # Save data for future sessions
        save_rag_data(chunks, embeddings)
        
        # Complete the progress bar
        progress_bar.progress(100)
        time.sleep(0.5)  # Small delay for visual feedback
        
        # Clear the progress elements and show success message
        progress_placeholder.empty()
        status_text.empty()
        st.success(f"✅ Successfully processed document with {len(chunks)} text segments")
        return True
        
    except Exception as e:
        # In case of any error, clear the progress elements and show error
        progress_placeholder.empty()
        status_text.empty()
        st.error(f"Error processing PDF: {str(e)}")
        return False

# Display sample questions for user convenience
def display_sample_questions():
    st.markdown("### 🔍 Sample Questions")
    
    col1, col2 = st.columns(2)
    
    sample_questions = [
        "What is the concept of naturalness?",
        "How does naturalness relate to environmental sustainability?",
        "What are the key principles of natural design?",
        "How can naturalness be measured or quantified?",
        "What are examples of naturalness in everyday products?"
    ]
    
    def set_question(q):
        st.session_state.question = q
    
    with col1:
        for q in sample_questions[:3]:
            st.button(q, key=f"q_{q[:20]}", on_click=set_question, args=(q,))
    
    with col2:
        for q in sample_questions[3:]:
            st.button(q, key=f"q_{q[:20]}", on_click=set_question, args=(q,))

# Main application
def main():
    create_header()
    
    # Sidebar for configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/api-settings.png", width=50)
        st.title("Configuration")
        
        # API Key Input
        api_key_input = st.text_input(
            "Enter your OpenAI API key",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="Get your API key from https://platform.openai.com/api-keys",
            key="api_key_input"
        )
        
        if api_key_input:
            st.session_state["api_key"] = api_key_input
            # Display confirmation (for debugging)
            st.success("API key saved! (Key starts with: " + api_key_input[:4] + "...)")
            
        st.markdown("---")
        
        # Information about the document
        st.subheader("About the Document")
        st.markdown("""
        This application uses a predefined document about naturalness. 
        The document covers topics such as:
        
        - The concept of naturalness
        - Natural design principles
        - Sustainability and natural systems
        - Applications of naturalness in various fields
        """)
        
        st.markdown("---")
        
        # Information about how it works
        st.subheader("How it Works")
        st.markdown("""
        1. Your question is analyzed
        2. Relevant sections from the document are retrieved
        3. OpenAI's GPT-4 generates an answer based on those sections
        4. You can view the source text that informed the answer
        """)

    # Initialize processing - only needs to happen once
    if "is_processed" not in st.session_state:
        rag_data = load_rag_data()
        if rag_data:
            st.session_state["chunks"] = rag_data["chunks"]
            st.session_state["embeddings"] = rag_data["embeddings"]
            st.session_state["is_processed"] = True
        else:
            process_pdf()

    # Main content area
    st.markdown("### 💬 Ask a Question About Naturalness")
    
    # Display sample questions
    display_sample_questions()
    
    # Question input
    query = st.text_input(
        "Type your question here", 
        key="question",
        placeholder="E.g., What is the concept of naturalness?",
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask AI 🤖", type="primary")
    
    # Only process if we have a question and the user clicked the button
    if query and ask_button:
        if not st.session_state.get("api_key"):
            st.error("Please enter your OpenAI API key in the sidebar")
        elif not st.session_state.get("is_processed", False):
            st.error("Unable to process the document. Please check if the PDF file exists.")
        else:
            # Set the API key directly (to ensure it's always fresh)
            openai.api_key = st.session_state.get("api_key", "")
            
            # Create placeholders for search progress
            search_progress = st.empty()
            search_status = st.empty()
            
            # Get relevant chunks with visual progress
            search_progress_bar = search_progress.progress(0)
            search_status.text("Searching for relevant information...")
            
            time.sleep(0.5)  # Small delay for visual effect
            search_progress_bar.progress(50)
            
            relevant_chunks, similarities = get_relevant_chunks(
                query,
                st.session_state["chunks"],
                st.session_state["embeddings"],
                get_embedding_model()
            )
            
            search_progress_bar.progress(100)
            time.sleep(0.3)  # Small delay for visual effect
            
            # Clear search progress indicators
            search_progress.empty()
            search_status.empty()
            
            # Create placeholders for AI response generation
            response_progress = st.empty()
            response_status = st.empty()
            
            # Generate response with visual progress
            response_progress_bar = response_progress.progress(0)
            response_status.text("Generating response with AI...")
            
            # Simulate progress while actually generating the response
            # This gives better user feedback during the API call
            def update_progress():
                progress = 0
                while progress < 90:
                    time.sleep(0.1)
                    progress += 1
                    if progress % 3 == 0:  # Update less frequently to reduce flickering
                        response_progress_bar.progress(progress)
            
            # Start the progress bar animation in a separate thread
            thread = threading.Thread(target=update_progress)
            thread.start()
            
            # Actually generate the response
            response = generate_response(query, relevant_chunks)
            
            # Complete the progress bar
            response_progress_bar.progress(100)
            time.sleep(0.3)
            
            # Clear response progress indicators
            response_progress.empty()
            response_status.empty()
            
            # Display response in a nice container
            st.markdown("<div class='response-container'>", unsafe_allow_html=True)
            st.markdown("### 📝 Answer")
            st.write(response)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show sources
            with st.expander("📚 View Source Material"):
                st.markdown("The following sections from the document were used to generate the answer:")
                for i, (chunk, similarity) in enumerate(zip(relevant_chunks, similarities)):
                    st.markdown(f"<div class='source-container'>", unsafe_allow_html=True)
                    st.markdown(f"**Section {i+1}** (Relevance: {similarity:.2f})")
                    st.text(chunk[:300] + ("..." if len(chunk) > 300 else ""))
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("---")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>Powered by OpenAI's GPT-4 and Sentence Transformers • Made with Streamlit</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
