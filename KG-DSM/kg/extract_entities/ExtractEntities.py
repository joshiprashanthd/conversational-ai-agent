import sys, re

sys.path.append("../")

from typing import List
from groq_model import GroqModel
from kg.types import Entity


class ExtractEntities:
    def __init__(self, llm: GroqModel):
        self.llm = llm

    def build_prompt(self, text):
        self.prompt = """You are mental health expert. Your goal is to identify and categorize key entities from the given unstructured mental health literature text. 

Extract the following types of entities:
    - Symptoms: Terms describing mental or physical symptoms, such as 'sadness,' 'anxiety,' or 'fatigue.'
    - Disorders: Names of mental health conditions, such as 'depression,' 'ADHD,' 'PTSD,' or similar.
    - Psychometric Scores: Mentions of assessment scales or scores, such as 'PHQ-9' or 'GAD-7.'
    - Risk Factors: Factors that may increase severity of a symptom, such as 'family history,' 'genetics,' or 'past trauma.'
    - Etiological Factors: Factors that directly leading to a disorder, such as 'neurotransmitter imbalance,' 'environmental stressors,' or 'childhood abuse.'
    - Distal Factors: Factors that are indirectly related to a condition, such as 'socioeconomic status,' 'cultural factors,' or 'climate.'
    - Treatments: A treatment for a specific condition, such as "chemotherapy" for "cancer" or "cognitive behavioral therapy" for "depression"
    - Relievers: Provides temporary relief from symptoms, such as 'painkillers,' 'relaxation techniques,' or "deep breathing" or 'antidepressants' for 'depression.' 
    - Prevention: Methods or practices that prevent the occurrence of a condition, such as 'vaccination,' 'healthy diet,' or 'exercise.'
    - Population Groups: Groups of people with shared characteristics, such as 'children,' 'elderly,' 'veterans,' or 'students.'

Task Instructions:
    - Remember that these entities are supposed to be used for knowledge graph construction.
    - Give a brief description of the entity in the context of the text.
    - If an entity can be classified into multiple types, list all possible types separated by commas.
    - If an entity is ambiguous, provide a brief explanation of the ambiguity. Put explanations in parentheses in descriptions.

Output Format:
Entity: <entity text>
Description: <description of entity given in the text>
Type: <entity type, multiple types separated by comma if entity is ambiguous>


### LITERATURE TEXT START ###
{text}
### LITERATURE TEXT END ###

Generate entities, descriptions, and types based on the given text.
Entity: """

        self.prompt = self.prompt.format(text=text)

    def parse_response(self, response) -> List[Entity]:
        pattern = re.compile(r"Entity: (.+?)\nDescription: (.+?)\nType: (.+?)\n")
        matches = pattern.findall(response)
        entities = []
        for match in matches:
            entity, description, entity_type = match
            entity_type = entity_type.split(", ")
            entity = Entity(entity=entity, description=description, type=entity_type)
            entities.append(entity)
        return entities

    def __call__(self, text: str) -> List[Entity]:
        self.build_prompt(text)
        response = self.llm.completion(self.prompt)
        return self.parse_response(response)
