"""
Utility modules for the OmniQA chatbot.
"""
from .document_processor import load_docs_from_folder, process_uploaded_file

__all__ = [
    'load_docs_from_folder',
    'process_uploaded_file'
]