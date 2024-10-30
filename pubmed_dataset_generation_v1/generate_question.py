import json
import os
from pubmed_dataset_generation_v1.QA_prompt import prompt_QA
from pubmed_dataset_generation_v1.validate_prompt import validate_topic
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from lib.groq_model import Llama3_1_8B_Instant

model = Llama3_1_8B_Instant()


def generate_questions(title, abstract):
    prompt = prompt_QA(title, abstract)
    response = model.completion(prompt)
    return response


def parse_llm_response(response):
    """
    Parse the plain text response from LLM into a structured format.
    """
    parsed_question = {}

    # Use regular expressions to find the relevant parts of the response
    question_match = re.search(r"- Question: (.*)", response)
    optA_match = re.search(r"- optA: (.*)", response)
    optB_match = re.search(r"- optB: (.*)", response)
    optC_match = re.search(r"- optC: (.*)", response)
    optD_match = re.search(r"- optD: (.*)", response)
    correct_answer_match = re.search(r"- Correct Answer: (.*)", response)
    explanation_match = re.search(r"- Explanation: (.*)", response)
    subject_match = re.search(r"- Subject: (.*)", response)
    topic_match = re.search(r"- Topic: (.*)", response)

    # Populate parsed_question dictionary with extracted values
    parsed_question["question"] = question_match.group(1) if question_match else ""
    parsed_question["optA"] = optA_match.group(1) if optA_match else ""
    parsed_question["optB"] = optB_match.group(1) if optB_match else ""
    parsed_question["optC"] = optC_match.group(1) if optC_match else ""
    parsed_question["optD"] = optD_match.group(1) if optD_match else ""
    parsed_question["correctAnswer"] = (
        correct_answer_match.group(1) if correct_answer_match else ""
    )
    parsed_question["explanation"] = (
        explanation_match.group(1) if explanation_match else ""
    )
    parsed_question["subject"] = subject_match.group(1) if subject_match else ""
    parsed_question["topic"] = topic_match.group(1) if topic_match else ""

    return parsed_question


# Load the abstracts
file_path = "sample_abstracts.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Generate questions and distractors for each abstract
output = {}
no = 0
yes = 0
# Main loop to process abstracts and generate questions
for idx, abstract in enumerate(data):
    title = abstract["title"]
    text = abstract["abstract"]
    pmid = abstract.get("pmid", "")

    # Generate QA using the LLM
    response = generate_questions(title, text)

    # Parse the LLM's plain text response to dict
    questions_parsed = parse_llm_response(response)

    # validating
    validation_response = validate_topic(
        subject=questions_parsed["subject"], topic=questions_parsed["topic"]
    )

    if questions_parsed["question"] == "":
        no += 1
        continue

    if validation_response.lower() == "no":
        no += 1
        continue

    # Store the parsed questions etc in output
    yes += 1
    output[idx] = {
        "pmid": pmid,
        # "title": title,
        # "abstract": text,
        "questions": [questions_parsed],  #
    }

# Save output to json
output_file = "generated_questions.json"
print(f"Generated questions for {yes} abstracts and skipped {no} abstracts.")
with open(output_file, "w") as file:
    json.dump(output, file, indent=4)
