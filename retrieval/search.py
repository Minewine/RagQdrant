# retrieval/search.py
from vectorization.embedding_processing import embedding_model, generate_embeddings

def process_query(query):
    """Process the user query (e.g., clean text, generate query embedding)."""
    # Simple preprocessing: strip or lower-case if needed
    cleaned_query = query.strip()
    # Generate an embedding for semantic search
    try:
        query_vec = embedding_model.encode([cleaned_query])[0]
    except Exception:
        query_vec = None
    return cleaned_query, query_vec

def search_vectors(query_vector, top_k=5):
    """Search the vector database for nearest vectors to the query vector."""
    results = []
    if query_vector is None:
        return results
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(":memory:")  # This should connect to the same DB used in storage
        # Perform a vector similarity search
        search_result = client.search(collection_name="documents", query_vector=query_vector, limit=top_k)
        results = [hit.payload.get("id") for hit in search_result]  # get IDs of matching chunks
    except Exception as e:
        from utils.logging_handler import log_activity
        from utils.error_handler import handle_errors
        log_activity("Vector search failed.")
        handle_errors(e)
    return results

def search_keywords(query, documents):
    """Fallback keyword search through documents (or their text) for the query."""
    matches = []
    for doc_id, text in documents.items():
        if query.lower() in text.lower():
            matches.append(doc_id)
    return matches

def re_rank_results(vector_results, keyword_results):
    """Combine and re-rank results from vector and keyword searches."""
    # Simple strategy: merge lists (vector results first) and remove duplicates
    combined = []
    for rid in vector_results + keyword_results:
        if rid not in combined:
            combined.append(rid)
    return combined

def handle_table_queries(query, tables):
    """If the query likely refers to tabular data, search within extracted tables."""
    # Example: if query contains numeric patterns or keywords like "table" or "data"
    relevant_tables = []
    if "table" in query.lower() or any(char.isdigit() for char in query):
        for idx, table in enumerate(tables):
            # simple check: if any cell in table contains query substring
            table_text = "\n".join(["\t".join(row) for row in table])
            if query.lower() in table_text.lower():
                relevant_tables.append((idx, table))
    return relevant_tables
