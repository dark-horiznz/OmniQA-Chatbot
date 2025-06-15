from .gemini_scraper import run
import warnings
warnings.filterwarnings("ignore")

def run_scraping(question, context='', debug=False):
    """
    Run the scraping process for a given question and context.

    Args:
        question (str): The user's question.
        context (str): Additional context to consider.
        debug (bool): If True, prints debug information.

    Returns:
        dict: Results of the scraping process.
    """
    return run(question, context, debug)