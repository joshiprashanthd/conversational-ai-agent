from groq_model import Groq
from typing import TypedDict, List, Dict
import re

class QNA(TypedDict):
    question: str
    answer: str

class Response(TypedDict):
    parameters: Dict[str, List[str]]

class ExtractL1Parameter:
    def __init__(self, llm: Groq) -> None:
        self.llm = llm

    def build_prompt(self, answers: List[QNA], parameters: List[str]) -> None:
        self.prompt = """You are an expert mental health professional. Your goal is to read patient's answers to some of the questions given and find out the parameter they belong to and extract relevant phrases from the answer which can be useful to assess patient's health state.

Task Instructions:
    - You can pick multiple parameters which are most suitable for the answer.
    - Parameters for each question are provided, do not create parameters of your own.
    - Always add number and the title of the question before generating parameters.
    - Do not repeat answer in your response.
    - Enclose all the comma separated parameters you generate with left bracket [ and right bracket ]
    - Enclose each of the comma separated extracted phrases in left angle bracket < and right angle bracket >.

parameters:
{parameters}

{qnas}

Extract relevant phrases from the answers and assign parameters from the given list for each answer:
"""

        qna_template = """
Question: {question}
Answer: {answer}
"""

        qnas = []
        for i, d in enumerate(answers):
            qna = qna_template.format(
                question=d["question"],
                answer=d["answer"],
            )
            qnas.append(qna)

        qnas_str = "\n".join(qnas)
        self.prompt = self.prompt.format(qnas=qnas_str, parameters=", ".join(parameters))

    def extract_strings(self, input_string: str, encloser: str) -> List[str]:
        enclosers = {"(": ")", "[": "]", "{": "}", "<": ">"}

        if encloser not in enclosers:
            raise ValueError("Invalid encloser. Supported enclosers are (, [, {, and <")

        closing_encloser = enclosers[encloser]

        escaped_open = re.escape(encloser)
        escaped_close = re.escape(closing_encloser)

        pattern = escaped_open + r"(.*?)" + escaped_close
        return re.findall(pattern, input_string)

    def process_response(self, response: str) -> Response:
        parameters_list = self.extract_strings(response, "[")
        phrases_list = self.extract_strings(response, "<")

        # Ensure both lists are of the same length
        if len(parameters_list) != len(phrases_list):
            raise ValueError("Mismatched number of parameters and phrases.")

        # Create a dictionary to map parameters to phrases
        parameters_dict = {}
        
        for param_str, phrase_str in zip(parameters_list, phrases_list):
            params = [param.strip() for param in param_str.split(',')]
            phrases = [phrase.strip() for phrase in phrase_str.split(',')]
            
            for param in params:
                if param not in parameters_dict:
                    parameters_dict[param] = []
                parameters_dict[param].extend(phrases)

        return {"L1 parameters dict": parameters_dict}

    def __call__(self, answers: List[QNA], parameters: List[str]) -> Response:
        self.build_prompt(answers, parameters)
        response = self.llm.completion(self.prompt)
        return self.process_response(response)
