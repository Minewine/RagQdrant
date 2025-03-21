from ingestion.qdrant_utils import store_documents, query_chatbot, init_collection

# Initialize Qdrant collection
init_collection()

# Store a sample document
file_path = '/home/mick/Projects/RagQdrant/docs'
store_documents(file_path, doc_id="eurostat")

# Query the chatbot
response = query_chatbot("What is in the document?")
print(response)
