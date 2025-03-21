import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def log_activity(message):
    """Log a general activity or event in the RAG system."""
    logging.info(message)