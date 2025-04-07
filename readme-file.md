# Claude RAG System for Naturalness PDF Documents

This application allows you to upload a PDF about naturalness and use Claude's AI capabilities to answer questions about the content. It implements a Retrieval-Augmented Generation (RAG) system that:

1. Extracts text from your PDF
2. Breaks it into manageable chunks
3. Creates vector embeddings for semantic search
4. Retrieves the most relevant chunks for any question
5. Uses Claude to generate answers based only on the retrieved content

## Features

- PDF text extraction and processing
- Vector embeddings for semantic search
- Contextual question answering with Claude
- Simple, intuitive Streamlit interface
- Persistent storage of processed documents
- Transparent source references

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- An Anthropic API key (get one at https://console.anthropic.com/)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd claude-naturalness-rag
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

4. Open your browser and go to http://localhost:8501

### Usage

1. Enter your Anthropic API key in the sidebar
2. Upload a PDF document about naturalness
3. Click "Process PDF" to extract and index the content
4. Enter your questions in the text input
5. View Claude's answers and the source chunks that informed them

## How It Works

1. **Text Extraction**: PyPDF extracts text from your uploaded document.
2. **Chunking**: The text is divided into overlapping chunks with intelligent paragraph breaks.
3. **Embedding**: Sentence Transformers creates vector embeddings for each chunk.
4. **Retrieval**: When you ask a question, the system finds the most semantically similar chunks.
5. **Generation**: Claude uses the retrieved chunks as context to answer your question.

## Performance Notes

- The quality of answers depends on the content and clarity of your PDF.
- Sentence transformer embeddings provide efficient semantic search capabilities.
- Claude is instructed to only use information from the retrieved chunks.
- The system uses the claude-3-5-sonnet model for optimal performance.

## Customization

You can modify these parameters in the code:
- `EMBEDDING_MODEL`: The sentence transformer model used for embeddings
- `CHUNK_SIZE`: The size of text chunks (in characters)
- `CHUNK_OVERLAP`: The overlap between chunks
- `MAX_CHUNKS_TO_USE`: The number of chunks to retrieve for each question

## License

This project uses open-source components with their respective licenses.
