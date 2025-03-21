# utils/error_handler.py
import logging
def handle_errors(error):
    """Handle errors by logging them and (optionally) raising exceptions."""
    logging.error(f"Error: {error}")
    # In a real system, you might perform additional steps like notifying a monitoring service.
    # For now, just print the error message.
    print(f"An error occurred: {error}")