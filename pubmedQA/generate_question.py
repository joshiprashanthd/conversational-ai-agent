import json
import os
from QA_prompt import prompt_QA
from validate_prompt import validate_abstract
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.groq_model import (
    Llama3_1_8B_Instant,
    Llama3_1_70B_Versatile,
    Llama3_1_405B_Reasoning,
)


model = Llama3_1_8B_Instant()

def generate_questions(title, abstract):
    prompt = prompt_QA(title, abstract)
    response = model.completion(prompt)
    return response

# Load the abstracts
file_path = "sample_abstracts.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Generate questions and distractors for each abstract
output = {}
for idx, abstract in enumerate(data):
    title = abstract["title"]
    text = abstract["abstract"]
    # Validate the abstract before generating questions
    validation_response = validate_abstract(title, text)
    # prune the validation response to get 'yes' or 'no'
    if validation_response == "No":
        output[idx] = {"title": title, "abstract": text, "questions": []}
        continue
    response = generate_questions(title, text)
    output[idx] = {"title": title, "abstract": text, "questions": response}

# Save the output to a JSON file
output_file = "generated_questions.json"
with open(output_file, "w") as file:
    json.dump(output, file, indent=4)
