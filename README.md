# Claude RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline powered by Claude Sonnet 4 for intelligent document question-answering.

## Features

- **Multi-format document support**: PDF, DOCX with intelligent text extraction
- **Specialized utility bill processing**: Enhanced OCR for challenging utility bill documents
- **Vector similarity search**: Using ChromaDB and sentence transformers
- **Claude Sonnet 4 integration**: High-quality response generation
- **Multiple interfaces**: Both CLI and web interfaces available
- **Evaluation framework**: Built-in testing and evaluation capabilities

## Architecture

### Core Components

- **`rag_pipeline.py`**: Main orchestrator for document ingestion and querying
- **`document_processor.py`**: Handles PDF/DOCX text extraction with OCR capabilities
- **`vector_store.py`**: Manages ChromaDB for document embeddings and similarity search
- **`utility_bill_processor.py`**: Specialized processor for utility bills with enhanced OCR

### Interfaces

- **`cli.py`**: Command-line interface for batch operations
- **`web_app.py`**: Streamlit web application for interactive use
- **`evaluate.py`**: Evaluation framework for testing pipeline performance

## Setup

### Prerequisites

- Python 3.8+
- Tesseract OCR (for utility bill processing)
- Anthropic API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gauravc/claude-rag-pipeline.git
cd claude-rag-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Environment Variables

Create a `.env` file with the following variables:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
CHUNK_SIZE=500
CHUNK_OVERLAP=50
USE_OCR=true
CHROMA_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Usage

### Command Line Interface

1. **Ingest documents**:
```bash
python src/cli.py ingest --path /path/to/documents
```

2. **Query documents**:
```bash
python src/cli.py query "What is the total energy cost?"
```

### Web Interface

Launch the Streamlit web application:

```bash
streamlit run src/web_app.py
```

Then open your browser to `http://localhost:8501`

### Python API

```python
from src.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest documents
pipeline.ingest_documents("./documents")

# Query
result = pipeline.query("What are the main topics discussed?")
print(result['answer'])
```

## Document Support

### Supported Formats
- PDF files (with OCR fallback)
- DOCX files
- Specialized utility bill processing

### Utility Bills
The pipeline includes enhanced processing for utility bills with:
- High-resolution image conversion
- Multiple OCR techniques
- Table detection and extraction
- Structured information extraction

## Testing

Run the test suite:

```bash
python src/test_integration.py
python src/test_utility_processor.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **OCR not working**: Ensure Tesseract is installed and in your PATH
2. **API errors**: Verify your Anthropic API key is correct and has sufficient credits
3. **Memory issues**: Reduce batch size in vector store operations

### Support

For issues and questions, please open an issue on GitHub.
