import sys, re

sys.path.append("../")

from Dataset import Publication
from typing import List
from groq_model import GroqModel


class MentalHealthCategory:
    def __init__(self, llm: GroqModel) -> None:
        self.llm = llm

    def build_prompt(self, publications: List[Publication]):
        self.prompt = """You are a medical expert. Your goal is to identify the mental health disorder an abstract of a research paper talks about.

Mental health category we are interested are Stress, Trauma, Anxiety and Depression.

Task Instruction:
    - You will be given an id, title and abstract of the research paper.
    - Carefully read the abstract of the research paper.
    - Do your best to identify relevant keywords that could help identify the mental health category.
    - If keywords does not help then use your best judgement to decide mental health category.
    - An abstract may belong to multiple mental health categories.
    - If abstract is not provided or abstract is not related to medicine or abstract is not related any of the above mentioned mental health categories then assign label "None"

Output Format:
Id: <id of the research paper>
Label: <comma separated mental health categories>

Here are the ids, titles and abstracts of the research paper. Carefull read each abstract and assign label to it.
{id_title_abstract}

Output: """
        id_title_abstract = []
        TEMP = """Id: {id}
Title: {title}
Abstract: {abstract}
"""
        for pub in publications:
            id_title_abstract.append(
                TEMP.format(id=pub.pmid, title=pub.title, abstract=pub.abstract)
            )

        id_title_abstract = "\n".join(id_title_abstract)

        self.prompt = self.prompt.format(id_title_abstract=id_title_abstract)

    def parse_response(self, response):
        pattern = r"^Id:\s+(\d+)\s*\nLabel:\s*(.+)\s*$"
        matches = re.findall(pattern, response, re.MULTILINE)
        data = [{"pmid": int(id), "label": label} for id, label in matches]
        return data

    def __call__(self, publications: List[Publication]):
        self.build_prompt(publications)
        response = self.llm.completion(self.prompt)
        return self.parse_response(response)
