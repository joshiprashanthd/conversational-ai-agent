from groq_model import Groq
from typing import TypedDict, List, Dict, Tuple
import re

class QNA(TypedDict):
    question: str
    answer: str

class Response(TypedDict):
    parameters: Dict[str, List[str]]

class ExtractL3Parameter:
    def __init__(self, llm: Groq) -> None:
        self.llm = llm

    def build_prompt(self, phrases: List[str], detailed_parameters: List[str]) -> None:
        self.prompt = """You are an expert mental health professional. Your goal is to read the list of phrases and find out the detailed parameters that best describe each phrase.

Task Instructions:
    - You need to assign each phrase to a detailed parameter from the given list.
    - Enclose each phrase with left angle bracket < and right angle bracket >.
    - Enclose each detailed parameter with left bracket [ and right bracket ].

Detailed parameters:
{detailed_parameters}

Phrases:
{phrases}

Assign each phrase to the most suitable detailed parameter from the given list:
"""

        phrase_template = "<{phrase}>"

        phrases_str = "\n".join([phrase_template.format(phrase=phrase) for phrase in phrases])
        self.prompt = self.prompt.format(detailed_parameters=", ".join(detailed_parameters), phrases=phrases_str)

    def extract_strings(self, input_string: str, encloser: str) -> List[str]:
        enclosers = {"(": ")", "[": "]", "{": "}", "<": ">"}

        if encloser not in enclosers:
            raise ValueError("Invalid encloser. Supported enclosers are (, [, {, and <")

        closing_encloser = enclosers[encloser]

        escaped_open = re.escape(encloser)
        escaped_close = re.escape(closing_encloser)

        pattern = escaped_open + r"(.*?)" + escaped_close
        return re.findall(pattern, input_string)

    def process_response(self, response: str) -> List[Tuple[str, List[str]]]:
        parameters_list = self.extract_strings(response, "[")
        phrases_list = self.extract_strings(response, "<")

        # Initialize an empty list for results
        results = []

        # Iterate through phrases and parameters
        for phrase_str in phrases_list:
            phrase = phrase_str.strip()
            # Find the associated parameters
            param_list = []
            for param_str in parameters_list:
                param_list.extend([param.strip() for param in param_str.split(',')])
            results.append((phrase, param_list))

        return results

    def __call__(self, phrases: List[str], detailed_parameters: List[str]) -> List[Tuple[str, List[str]]]:
        self.build_prompt(phrases, detailed_parameters)
        response = self.llm.completion(self.prompt)
        return self.process_response(response)
