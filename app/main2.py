import os
import sys
import json
import requests
import spacy
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from ollama_backend import generate_response_with_stream

# Configurable FAISS store directory
FAISS_STORE_DIR = os.getenv("FAISS_STORE_DIR", "/data")
faiss_store_file = os.path.join(FAISS_STORE_DIR, "faiss_index.bin")
metadata_store_file = os.path.join(FAISS_STORE_DIR, "metadata_store.json")

# Initialize components
nlp = spacy.load("en_core_web_sm")  # Load spaCy for NLP tasks
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight embedding model
dimension = 384  # Embedding size for the model (matches MiniLM)
faiss_index = faiss.IndexFlatL2(dimension)  # FAISS vector store for semantic search
metadata_store = {}  # Metadata for source tracking
OLLAMA_PRIMARY_URL = "http://localhost:11434"
OLLAMA_FALLBACK_URL = "http://127.0.0.1:11434"
ollama_url = OLLAMA_PRIMARY_URL


def check_ollama_available():
    """Check if the Ollama service is reachable, fallback to localhost if needed."""
    global ollama_url
    for url in [OLLAMA_PRIMARY_URL, OLLAMA_FALLBACK_URL]:
        try:
            # Use a basic GET request to validate connectivity
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                print(f"Ollama service is available at {url}.")
                ollama_url = url
                return True
        except requests.ConnectionError:
            continue
    print("Error: Ollama service is not available. Please start Ollama and try again.")
    return False


def load_faiss_store():
    """Load the FAISS store and metadata if they exist."""
    global faiss_index, metadata_store
    if os.path.exists(faiss_store_file) and os.path.exists(metadata_store_file):
        print(f"Loading FAISS index and metadata from {FAISS_STORE_DIR}...")
        faiss_index = faiss.read_index(faiss_store_file)
        with open(metadata_store_file, "r") as f:
            metadata_store = json.load(f)
    else:
        print("No existing FAISS index found. Starting fresh.")


def save_faiss_store():
    """Save the FAISS store and metadata to disk."""
    os.makedirs(FAISS_STORE_DIR, exist_ok=True)
    print(f"Saving FAISS index and metadata to {FAISS_STORE_DIR}...")
    faiss.write_index(faiss_index, faiss_store_file)
    with open(metadata_store_file, "w") as f:
        json.dump(metadata_store, f)


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return ""


def extract_text_from_json(json_file):
    """Extract text from a JSON file."""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        return data.get("text", "")
    except Exception as e:
        print(f"Error extracting text from JSON {json_file}: {e}")
        return ""


def extract_text_from_text_file(text_file):
    """Extract text from a plain text file."""
    try:
        with open(text_file, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Error extracting text from text file {text_file}: {e}")
        return ""


def chunk_text(text):
    """Hierarchical text chunking."""
    doc = nlp(text)

    # Topic-Level Chunking
    sections = [chunk.text for chunk in doc.sents if len(chunk.text) > 50]

    # Sentence-Level Chunking
    sentences = [sent.text for sent in doc.sents]

    # Semantic-Level Chunking
    semantic_chunks = []
    current_chunk = []
    max_chunk_size = 500  # Max characters in a chunk
    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk)) > max_chunk_size:
            semantic_chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        semantic_chunks.append(" ".join(current_chunk))
    return sections, sentences, semantic_chunks


def process_and_store_text(text, file_path):
    """Process a single document's text."""
    _, _, semantic_chunks = chunk_text(text)
    for idx, chunk in enumerate(semantic_chunks):
        embedding = embedding_model.encode(chunk)
        faiss_index.add(np.array([embedding]))
        metadata_store[faiss_index.ntotal - 1] = {
            "file_path": file_path,
            "chunk": chunk,
            "chunk_index": idx,
        }


def process_input_files(input_files):
    """
    Process input files based on their type.
    Args:
        input_files (list): List of file paths to process.
    """
    for file_path in input_files:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        print(f"Processing file: {file_path}")
        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif ext == ".json":
            text = extract_text_from_json(file_path)
        elif ext in [".txt", ".log"]:
            text = extract_text_from_text_file(file_path)
        else:
            print(f"Unsupported file type: {file_path}. Skipping...")
            continue

        process_and_store_text(text, file_path)


def retrieve(query):
    """Retrieve the most relevant chunks based on the query."""
    query_embedding = embedding_model.encode(query)
    distances, indices = faiss_index.search(np.array([query_embedding]), k=5)

    results = []
    for idx in indices[0]:
        metadata = metadata_store.get(idx)  # Safely get metadata
        if metadata:  # Only append if metadata exists
            results.append({"chunk": metadata.get("chunk"), "source": metadata.get("file_path")})
        else:
            print(f"Warning: No metadata found for index {idx}. Skipping...")
    return results


def generate_response(query):
    """Generate a response using Ollama as the LLM backend."""
    retrieved_chunks = retrieve(query)
    if not retrieved_chunks:
        return "No relevant information found for your query."

    sources = "\n".join(
        f"- {res['source']} (Excerpt: {res['chunk'][:100]}...)" for res in retrieved_chunks
    )
    context = "\n".join(res["chunk"] for res in retrieved_chunks)
    response = generate_response_with_stream(query, context, ollama_url)
    return f"Query: {query}\n\nSources:\n{sources}\n\nResponse:\n{response}"


if __name__ == "__main__":
    # Check Ollama availability
    if not check_ollama_available():
        sys.exit(1)

    # Load FAISS index if it exists
    load_faiss_store()

    # Expect at least one input file
    input_files = sys.argv[1:]
    if input_files:
        process_input_files(input_files)
        save_faiss_store()
    else:
        print("No input files provided. Skipping ingestion.")

    # Interactive querying
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break
        print(generate_response(query))

