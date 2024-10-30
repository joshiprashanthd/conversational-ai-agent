from pubmed_dataset_generation.Dataset import Publication
from typing import List
from lib.groq_model import GroqModel


class MedicineCategory:
    def __init__(self, llm: GroqModel) -> None:
        self.llm = llm

    def build_prompt(self, publications: List[Publication]):
        self.prompt = """You are a medical expert. Your goal is to identify whether an abstract of a research paper talks about modern medicine or traditional indian medicinal practices.

Traditional Indian Medicinal Practices consist of Ayurveda, Yoga, Siddha, Unani, Homeopathy, Naturopathy, Panchakarma, Siddha Marmam.
    
Task Instruction:
    - You will be given an id, title and abstract of the research paper.
    - Carefully read the abstract of the research paper.
    - Do your best to identify relevant keywords that belong to modern medicine and traditional indian medicinal practices.
    - If most of the keywords in the abstract belong to modern medicine then assign label "Modern Medicine"
    - If most of the keywords in the abstract belong to tranditional indian medicinal practices then assign label "Traditional Indian Medicine"
    - If abstract is not provided or abstract is not related to medicine then assign label "None"

Output Format:
Id: <id of the research paper>
Label: <label of research paper>

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
