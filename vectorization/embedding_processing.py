# vectorization/embedding_processing.py
from sentence_transformers import SentenceTransformer

# Initialize a global or cached model for embeddings (e.g., all-MiniLM-L6-v2)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def split_into_chunks(text, chunk_size=500, overlap=50):
    """Split text into chunks of approximately chunk_size characters, with overlap."""
    chunks = []
    if not text:
        return chunks
    # Simple split by length (could also split by sentences or use LangChain's TextSplitter)
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def generate_embeddings(chunks):
    """Generate vector embeddings for each text chunk using a pretrained model."""
    try:
        vectors = embedding_model.encode(chunks)  # returns a list/array of vector embeddings
        return vectors
    except Exception as e:
        from utils.logging_handler import log_activity
        from utils.error_handler import handle_errors
        log_activity("Embedding generation failed.")
        handle_errors(e)
        return []

def store_embeddings(vectors, ids=None):
    """Store embeddings in a vector database or other storage for retrieval."""
    # Example using Qdrant (vector database)
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(":memory:")  # using in-memory for example
        collection_name = "documents"
        client.recreate_collection(collection_name=collection_name, vector_size=len(vectors[0]), distance="Cosine")
        payloads = [{"id": idx} for idx in range(len(vectors))] if ids is None else [{"id": id} for id in ids]
        client.upsert(
            collection_name=collection_name,
            points=[{"id": payload["id"], "vector": vec} for vec, payload in zip(vectors, payloads)]
        )
        return True
    except Exception as e:
        from utils.logging_handler import log_activity
        from utils.error_handler import handle_errors
        log_activity("Storing embeddings failed.")
        handle_errors(e)
        return False
