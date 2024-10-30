import re

from Dataset import Publication
from typing import List
from groq_model import GroqModel


class OnePrompt:
    def __init__(self, llm: GroqModel):
        self.llm = llm

    def build_prompt(self, publications: List[Publication]):
        self.prompt = """Role: You are an advanced language model with the role of generating questions from research abstracts in the field of mental health. Your questions will cover diverse types—factoid, reasoning, and yes/no—derived from details found within each abstract. These questions are intended for use in a knowledge graph, so it’s crucial they capture significant information and highlight advanced concepts.

Goal: The objective is to create a single, most insightful question from each abstract, incorporating as many relevant details as possible. This question will help users engage deeply with the content, aiding comprehension and facilitating the construction of a knowledge graph.

Task Instructions:
1. Identify Question Type: 
    - Assess if a question can be derived from the abstract.
    - Choose the best question type from the following:
        - Factoid: A question with a concrete answer. Structure this as a multiple-choice question (MCQ) with one correct answer and three plausible distractors.
        - Reasoning: A question that prompts an explanation, often beginning with "why." These questions should reveal causal relationships or concepts.
        - Yes or No: A binary question, often beginning with "is," "does," or "can."

2. Question Construction:
    - Select statements from the abstract that contain essential information but hide the actual answer. Use all other available words to structure the question.
    - Aim for advanced-level questions that challenge the reader’s understanding of complex ideas within the abstract.
    - Ensure the question is not overly specific to the knowledge present in the research paper, but rather captures broader concepts and insights that can be generalized.

3. Categories and Edge Cases:
    - Focus on mental health or related medical topics.
    - If the abstract does not lend itself easily to one question type, prioritize reasoning-type questions for their explanatory potential, or choose factoid-type if specific information is available.
    - Handle ambiguous or complex terms by rephrasing for clarity without diluting scientific accuracy.
    
Output Format:

Id: Id of the research paper
Valid Question: True if question can be derived from the abstract otherwise False
Type of Question: Factoid, Reasoning, or Yes/No
Question: The formulated question following the chosen type
Options: Comma separated list of 4 options. One correct option and three plausible distractors. Enclose each option in left angle bracket < and right angle bracket >. Put N/A if question type is not Factoid
Correct Answer: Answer to the prepared question. Follow the following rules for the format of this field:
	- Yes or No if question type if Yes/No
	- Reasoning for the prepared question if question type is reasoning
	- Correct option from the 4 options if question type is Factoid

Here are the abstracts of the research papers. Read each abstract and generate a question based on the instructions provided.
{abstracts}

Id: """

        ABSTRACT = """Id: {pmid}
Abstract: {abstract}
"""

        for pub in publications:
            pub.abstract = pub.abstract.replace("\n", " ")
            pub.abstract = pub.abstract.replace("Abstract", "")
            pub.abstract = pub.abstract.strip()
            pub.abstract = ABSTRACT.format(pmid=pub.pmid, abstract=pub.abstract)

        abstracts = "\n".join([pub.abstract for pub in publications])
        self.prompt = self.prompt.format(abstracts=abstracts)

    def parse_response(self, response):
        pattern = r"Id:\s*(\d+)\nValid Question:\s*(True|False)\nType of Question:\s*(Factoid|Reasoning|Yes or No)\nQuestion:\s*([^\n]+)(?:\nOptions:\s*([^\n]+))?(?:\nCorrect Answer:\s*([^\n]+))?"

        matches = re.findall(pattern, response)

        parsed_data = []
        for match in matches:
            question_data = {
                "id": int(match[0]),
                "valid_question": match[1] == "True",
                "type_of_question": match[2],
                "question": match[3].strip(),
                "options": (
                    [option.strip() for option in re.findall(r"<(.*?)>", match[4])]
                    if match[4] and match[4] != "N/A"
                    else None
                ),
                "correct_answer": match[5].strip("<>"),
            }
            parsed_data.append(question_data)

        return parsed_data

    def __call__(self, publications: List[Publication]):
        self.build_prompt(publications)
        response = self.llm.completion(self.prompt)
        print("RESPONSE: ", response)
        return self.parse_response(response)
