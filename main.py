from pipeline.runner import run_main
import warnings
warnings.filterwarnings("ignore")

question = "What are the three types of machine learning?"
web_mode = False  # Disable web search to test local docs only

result = run_main(
    question,
    web_mode=web_mode
)

print(result)