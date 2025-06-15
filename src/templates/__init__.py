from .templates import make_templates
import warnings
warnings.filterwarnings("ignore")

def templates():
    """
    Create and return the necessary templates for the application.
    Returns:
        tuple: Contains the answer template, clarify template, summary template,
               web summary template, and final summary template.
    """
    return make_templates()