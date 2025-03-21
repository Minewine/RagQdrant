# RagQdrant

RagQdrant is a **Retrieval-Augmented Generation (RAG) system** powered by **Qdrant** for vector search and **Ollama** for LLM-based responses. It allows users to store, retrieve, and query documents efficiently using embeddings and semantic search.

## 🚀 Features
- **Document Ingestion**: Extracts and embeds text from PDFs, TXT, and DOCX files.
- **Vector Search**: Uses Qdrant as a high-performance vector database.
- **Query Answering**: Retrieves relevant document chunks and generates responses using an LLM.
- **Efficient Chunking & Embeddings**: Uses `all-MiniLM-L6-v2` for fast, lightweight embeddings.
- **Cross-Encoder Reranking**: Improves result quality with `cross-encoder/ms-marco-MiniLM-L-12-v2`.

## 📦 Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Minewine/RagQdrant.git
   cd RagQdrant
   ```
2. **Set up a Virtual Environment (Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Qdrant Locally** (if not using an external instance)
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## 📝 Usage
### **1️⃣ Initialize and Store Documents**
Modify `main.py` to process your documents:

```python
from ingestion.qdrant_utils import store_documents, query_chatbot, init_collection

init_collection()  # Ensure Qdrant collection is ready
store_documents("./docs")  # Upload all files in `docs` directory
```

Run:
```bash
python main.py
```

### **2️⃣ Query the Chatbot**
Ask a question based on the stored documents:
```python
response = query_chatbot("What does the document say about X?")
print(response)
```

## 📁 Project Structure
```
RagQdrant/
│── docs/                  # Sample documents (PDF, TXT, DOCX)
│── ingestion/             # Document processing & storage
│   ├── document_processing.py
│   ├── qdrant_utils.py
│── retrieval/             # Query and search
│   ├── search.py
│── utils/                 # Logging & error handling
│   ├── logging_handler.py
│   ├── error_handler.py
│── vectorization/         # Embedding processing
│── main.py                # Entry point
│── requirements.txt       # Dependencies
│── README.md              # This file
```

## 🛠 Future Improvements
- ✅ Support for more file formats (HTML, CSV)
- ✅ GUI or API for easier interaction
- ✅ Multi-document summarization

## 🤝 Contributing
1. Fork the repo & create a new branch.
2. Make your changes & run tests.
3. Submit a Pull Request.

## 📜 License
MIT License. See `LICENSE` for details.

---
💡 **Need Help?** Open an issue on [GitHub](https://github.com/Minewine/RagQdrant/issues).

