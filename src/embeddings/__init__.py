from .gemini_embeddings import GeminiEmbeddings
import warnings
warnings.filterwarnings("ignore")

def get_embedding_models():
    """
    Returns a dictionary of available embedding models.
    The dictionary maps model names to their respective classes.
    Returns:
        dict: A dictionary containing the available embedding models.
    """
    return {
        'GeminiEmbeddings': GeminiEmbeddings
        }