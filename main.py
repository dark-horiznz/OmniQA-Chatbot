from pipeline.runner import run_pipeline
import warnings
warnings.filterwarnings("ignore")

question = "I am having a light headache and slight dizziness. I have taken paracetamol but it is not helping. What should I do next?"
web_mode = False

result = run_pipeline(
        question,
        web_mode=web_mode
    )

print(result)