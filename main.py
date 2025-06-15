from pipeline.runner import run_main
import warnings
warnings.filterwarnings("ignore")

question = "I am having a light headache and slight dizziness. I have taken paracetamol but it is not helping. What should I do next?"
web_mode = True


result = run_main(
        question,
        web_mode=web_mode
    )

print(result)