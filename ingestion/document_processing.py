import docx
import fitz
import nltk
from nltk.tokenize import sent_tokenize
import tabula
from PIL import Image
import pytesseract
from utils.logging_handler import log_activity
from utils.error_handler import handle_errors

nltk.download('punkt')  # Ensure sentence tokenizer is available

def load_document(path):
    try:
        if path.endswith('.docx'):
            return docx.Document(path)
        elif path.endswith('.pdf'):
            return fitz.open(path)
        else:
            with open(path, 'rb') as f:
                return f.read()
    except Exception as e:
        log_activity(f"Failed to load document: {path}")
        handle_errors(e)
        return None

def extract_text(document):
    try:
        if isinstance(document, docx.document.Document):
            return "\n".join(para.text for para in document.paragraphs)
        elif isinstance(document, fitz.Document):
            return "\n".join(page.get_text() for page in document)
        elif isinstance(document, bytes):
            return document.decode('utf-8', errors='ignore')
        else:
            return ""
    except Exception as e:
        log_activity("Error extracting text from document.")
        handle_errors(e)
        return ""

def chunk_text(text, max_chunk_size=512):
    """Splits text into semantically meaningful chunks using sentence boundaries."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
