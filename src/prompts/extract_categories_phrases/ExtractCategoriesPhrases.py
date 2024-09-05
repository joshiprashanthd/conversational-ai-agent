from typing import TypedDict, List
from groq_model import GroqModel
import re


class QNA(TypedDict):
    parameter: str
    categories: List[str]
    question: str
    answer: str


class Response(TypedDict):
    parameter: str
    states: List[str]
    phrases: List[str]


class ExtractCategoriesPhrases:
    def __init__(self, llm: GroqModel) -> None:
        self.llm = llm

    def build_prompt(self, answers: List[QNA]):

        self.prompt = """You are an expert mental health professional. Your goal is to read patient's answers to some of the questions given and find out the category (or categories) they belong to and extract relevant phrases from the answer which can be useful to assess patient's mental state.

Task Instructions:
	- You can pick multiple categories which are most suitable for the answer.
	- Categories for each question are provided, do not create categories of your own.
    - All ways add number and the title of the question before generating categories.
    - Do not repeat answer in your response.
	- Enclose all the comma separated categories you generate with left bracket [ and right bracket ]
	- Enclose each the comma separated extracted phrases in left angle bracket < and right angle bracket >.
	

{qnas}

Extract relevant phrases from the answer and assign categories for each answer:
"""

        qna_template = """{num}. {parameter}
Categories: {categories}
Question: {question}
Answer: {answer}
"""

        qnas = []
        for i, d in enumerate(answers):
            categories_text = "[" + ", ".join(d["categories"]) + "]"
            qna = qna_template.format(
                num=i + 1,
                parameter=d["parameter"],
                categories=categories_text,
                question=d["question"],
                answer=d["answer"],
            )
            qnas.append(qna)

        qnas = "\n".join(qnas)

        self.prompt = self.prompt.format(qnas=qnas)

    def process_response(self, output: str) -> List[Response]:
        sections = re.split(r"\n\d+\.\s", output.strip())
        sections = [s.strip() for s in sections if s.strip()]

        result = []
        for section in sections:
            title_match = re.match(r"(.+?)(?:\n|$)", section)
            if not title_match:
                continue
            title = title_match.group(1).strip()

            square_bracket_content = re.findall(r"\[(.*?)\]", section)
            square_bracket_items = [
                item.strip()
                for item in ", ".join(square_bracket_content).split(",")
                if item.strip()
            ]

            curly_brace_content = re.findall(r"\<(.*?)\>", section)
            curly_brace_items = [
                item.strip()
                for item in ", ".join(curly_brace_content).split(",")
                if item.strip()
            ]

            res: Response = {
                "parameter": title,
                "states": square_bracket_items,
                "phrases": curly_brace_items,
            }
            result.append(res)

        return result

    def __call__(self, answers: List[QNA]):
        self.build_prompt(answers)
        response = self.llm.completion(self.prompt)
        print(response)
        responses = self.process_response(response)
        return responses
