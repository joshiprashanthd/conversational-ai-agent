from groq_model import GroqModel
import re
from typing import TypedDict

from Dataset import Publication


class MCQ(TypedDict):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str
    explanation: str


class GenerateQuestions:
    def __init__(self, llm: GroqModel):
        self.llm = llm

    def build_prompt(self, publication: Publication):
        title = publication.title
        abstract = publication.abstract
        self.prompt = """You are a highly knowledgeable medical expert with a deep understanding of research methodologies and medical terminologies. Your task is to generate multiple choice questions (MCQs) based on the abstract of a research paper. These questions will be used to test comprehension and critical thinking.

Task Instructions:
    - Carefully read the abstract word by word to fully understand the content.
    - Formulate a multiple choice question that accurately reflects the key points and findings of the abstract.
    - Ensure that the correct answer to the MCQ is explicitly stated within the abstract.
    - Create plausible distractors that are relevant to the content but clearly distinguishable from the correct answer.
    - Provide a detailed explanation for why the correct answer is accurate and why the distractors are incorrect.
    - Use the title of the paper to provide additional context and ensure the question aligns with the main theme of the research.


Output Format:
Question: The actual MCQ question based on the abstract.
Option A: The first option for the MCQ.
Option B: The second option for the MCQ.
Option C: The third option for the MCQ.
Option D: The fourth option for the MCQ.
Correct Answer: The correct option among A, B, C, or D.
Explanation: A detailed explanation of why the correct answer is accurate and why the distractors are incorrect.


Given the abstract below, write a multiple choice question based on the abstract.
Title: {title}
Abstract: {abstract}

""".format(
            title=title, abstract=abstract
        )

    def process_response(self, response) -> MCQ:
        fields = [
            "Question",
            "Option A",
            "Option B",
            "Option C",
            "Option D",
            "Correct Answer",
            "Explanation",
        ]

        parsed_question: MCQ = MCQ(
            question="",
            option_a="",
            option_b="",
            option_c="",
            option_d="",
            correct_answer="",
            explanation="",
        )

        for field in fields:
            match = re.search(rf"{field}: (.*)", response)
            field = field.replace(" ", "_").lower()
            parsed_question[field] = match.group(1) if match else ""
        return parsed_question

    def __call__(self, publication: Publication):
        self.build_prompt(publication)
        response = self.llm.completion(self.prompt)
        return self.process_response(response)
