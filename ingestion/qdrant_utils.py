from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer, CrossEncoder
from uuid import uuid4
from ingestion.document_processing import extract_text, chunk_text, load_document
from utils.logging_handler import log_activity
from utils.error_handler import handle_errors
from langchain_community.llms import Ollama


# 🔥 Initialize Qdrant client (Make sure it's running)
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

# 🔥 Define collection details
COLLECTION_NAME = "documents"
VECTOR_DIM = 384  # all-MiniLM-L6-v2 output size

ollama_llm = Ollama(model="qwen2.5:1.5b")


# 🔥 Ensure collection exists
def init_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
    )

# 🔥 Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔥 Load Cross-Encoder for reranking
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

import os
from pathlib import Path

def store_documents(file_paths, doc_id=None):
    """Stores documents from file or directory in Qdrant by extracting, chunking, and embedding them."""
    # 🔍 Normalize to list of paths
    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    all_files = []

    for path in file_paths:
        path = Path(path)
        if path.is_dir():
            all_files.extend([
                f for f in path.glob("**/*")
                if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".docx"}
            ])
        elif path.is_file():
            all_files.append(path)
        else:
            log_activity(f"[⚠️] Invalid path: {path}")

    for file_path in all_files:
        current_doc_id = doc_id or str(uuid4())
        document = load_document(str(file_path))

        if document is None:
            log_activity(f"⚠️ Failed to load document: {file_path}")
            continue

        text = extract_text(document)
        if not text.strip():
            log_activity(f"⚠️ No text extracted from {file_path}")
            continue

        chunks = chunk_text(text)
        embeddings = embedding_model.encode(chunks)

        metadata = {
            "filename": file_path.name,
            "doc_id": current_doc_id,
            "source": "Batch Upload",
        }

        points = [
            PointStruct(
                id=str(uuid4()),
                vector=emb.tolist(),
                payload={"text": chunk, "metadata": metadata}
            ) for chunk, emb in zip(chunks, embeddings)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        log_activity(f"✅ Stored {len(chunks)} chunks from {file_path}")

    print("✅ All documents processed and stored successfully!")


def query_chatbot(user_query, top_k=5):
    """Retrieves relevant text from multiple documents and generates a final response using Ollama."""
    try:
        # ✅ Step 1: Encode query
        query_embedding = embedding_model.encode(user_query)

        # ✅ Step 2: Search across all stored documents
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k * 2,  # Get more results to improve synthesis
            score_threshold=0.2
        )

        if not results:
            return "❌ No relevant information found."

        # ✅ Step 3: Extract relevant text from multiple documents
        rerank_inputs = [(user_query, res.payload["text"]) for res in results if "text" in res.payload]
        if not rerank_inputs:
            return "❌ No valid text retrieved from search results."

        scores = reranker.predict(rerank_inputs)
        ranked_results = sorted(zip(scores, results), reverse=True)

        # ✅ Step 4: Merge results across documents
        context_blocks = []
        seen_docs = set()
        
        for score, result in ranked_results[:top_k]:
            doc_metadata = result.payload["metadata"]
            doc_id = doc_metadata["doc_id"]
            text = result.payload["text"]

            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                context_blocks.append(f"📄 **Document:** {doc_metadata['filename']}\n{text}\n")

        final_context = "\n---\n".join(context_blocks)

        # ✅ Step 5: Send the combined context to Ollama
        final_prompt = f"""
        Based on the following information retrieved from multiple documents, answer the question in a detailed and well-structured way.

        **Context from Documents:**
        {final_context}

        **Question:** {user_query}
        """
        
        ollama_response = ollama_llm.invoke(final_prompt)

        # ✅ Step 6: Format and return response
        response = f"**🤖 AI Response:**\n{ollama_response}\n\n**📌 Retrieved from:**\n{final_context}"

        return response

    except Exception as e:
        return f"⚠️ Error in query_chatbot: {str(e)}"




