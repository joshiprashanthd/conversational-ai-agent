from lib.groq_model import GroqModel
from typing import TypedDict
import re


class TGenerateInteventionInput(TypedDict):
    parameter: str
    current_state: str
    target_state: str
    topic: str


class TGenerateInterventionResponse(TypedDict):
    reasoning: str
    activity: str


class GenerateIntervention:
    def __init__(self, llm: GroqModel):
        self.llm = llm

    def build_prompt(self, input: TGenerateInteventionInput):
        self.prompt = """We are gathering information about patient's sleep health. This information consist of various parameters each having some categories. A category describes the broad state of patient's sleep health in terms of the parameter.

Your goal is to make (reason, activity) pairs. An activity should be created such that it somehow changes the state of parameter from on category to another category.
Activities should be related to the topic given, for example if the topic is "yoga" then activity will have to contain some asanas that comes under yoga.

For example, suppose the category of parameter "thoughts before sleep" is "worried". This means that patient is worried about something before going to sleep. An activity like "go for a walk" could make the patient less "worried" and more "calm". This is how we can change the category of one parameter to other parameter by suggesting some activity.

Output Instructions:
    - Always generate a pair of reason and activity.
    - Always enclose reason in left angle bracket < and right angle bracket >.
    - Always enclose activity in left square bracket [ and right square bracket ].
    - Activity should be gentle enough to be done by an average person.
    - Reason should not be very generic, and be informative about the activity.

Given a parameter, current category and target category, suggest a (reason, activity) pair that changes the current category into target category.

Parameter: {parameter}
Current Category: {current_category}
Target Category: {target_category}
Topic: {topic}
Output:"""

        self.prompt = self.prompt.format(
            parameter=input["parameter"],
            current_category=input["current_state"],
            target_category=input["target_state"],
            topic=input["topic"],
        )

    def extract_strings(self, input_string, encloser):
        enclosers = {"(": ")", "[": "]", "{": "}", "<": ">"}

        if encloser not in enclosers:
            raise ValueError("Invalid encloser. Supported enclosers are (, [, {, and <")

        closing_encloser = enclosers[encloser]

        escaped_open = re.escape(encloser)
        escaped_close = re.escape(closing_encloser)

        pattern = f"{escaped_open}([^{escaped_open}{escaped_close}]*){escaped_close}"
        matches = re.findall(pattern, input_string)
        for match in matches:
            return match
        return ""

    def process_response(self, response: str) -> TGenerateInterventionResponse:
        return {
            "activity": self.extract_strings(response, "["),
            "reasoning": self.extract_strings(response, "<"),
        }

    def __call__(
        self, input: TGenerateInteventionInput
    ) -> TGenerateInterventionResponse:
        self.build_prompt(input)
        response = self.llm.completion(self.prompt)

        if not response:
            raise ValueError("No response from the model")

        return self.process_response(response)
