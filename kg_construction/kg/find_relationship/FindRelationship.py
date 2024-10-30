import re
from typing import List
from lib.groq_model import GroqModel
from kg_construction.kg.types import Entity, Relationship


class FindRelationship:
    def __init__(self, llm: GroqModel):
        self.llm = llm

    def build_prompt(self, entities: List[Entity]):
        self.prompt = """You are an expert in the field of knowledge graph and mental health. Your goal is to find the relationships between entities.

Task Instructions:
    - You will be given entity name, description, and type of it.
    - Use descriptions of entities to make a better decision in finding the relationships.

Strictly use these relations between entities:
1. ASSESSED_IN
    Source Entity Type: Symptom
    Target Entity Type: Metric
    (A symptom is measured or evaluated using a specific metric)

2. HAS_METRIC
    Source Entity Type: Disorder
    Target Entity Type: Metric
    (A disorder is measured or evaluated using a specific metric)

3. PREVENTS
    Source Entity Type: Prevention
    Target Entity Type: Disorder
    (A preventive measure helps avoid a disorder)

4. RELIEVES
    Source Entity Type: Reliever
    Target Entity Type: Symptom
    (A reliever reduces or alleviates symptoms)

5. CAUSES
    Source Entity Type: Etiological Factors
    Target Entity Type: Symptoms

    Source Entity Type: Etiological Factors
    Target Entity Type: Disorder
    (Etiological factors can directly cause symptoms or disorders)

6. MAY_CONTRIBUTE_TO
    Source Entity Type: Distal Factor
    Target Entity Type: Disorder
    (A distal factor might indirectly contribute to a disorder)

7. INCREASES_SEVERITY_OF
    Source Entity Type: Risk Factors
    Target Entity Type: Symptom
    (Risk factors can make symptoms worse)

8. TREATS
    Source Entity Type: Treatment
    Target Entity Type: Disorder
    (A treatment addresses a disorder)

9. INDICATES
    Source Entity Type: Symptom
    Target Entity Type: Disorders
    (A symptom can be indicative of certain disorders)

10. PREVALENT_IN
    Source Entity Type: Disorder
    Target Entity Type: Population Group
    (A disorder is more prevalent in specific population groups)

Output Format:
Source Entity: <name of the source entity>
Source Entity Type: <type of the source entity>
Relationship: <relationship between source and target entity>
Target Entity: <name of the target entity>
Target Entity Type: <type of the target entity>
Explanation: <explaination of relationship between source and target entity>

Here are the entities:
{entities}


Please find the relationships between the entities.

Source Entity: """

        ENTITY_TEMP = """Entity Name: {name}
Description: {description}
Entity Type: {type}
"""
        entities_text = []
        for entity in entities:
            entities_text.append(
                ENTITY_TEMP.format(
                    name=entity["entity"],
                    description=entity["description"],
                    type=entity["type"],
                )
            )

        self.prompt = self.prompt.format(entities="\n".join(entities_text))

    def parse_response(self, response: str):
        pattern = re.compile(
            r"Source Entity: (.+?)\nSource Entity Type: (.+?)\nRelationship: (.+?)\nTarget Entity: (.+?)\nTarget Entity Type: (.+?)\nExplanation: (.+?)\n"
        )
        matches = pattern.findall(response)
        relationships = []
        for match in matches:
            (
                source_entity,
                source_entity_type,
                relationship,
                target_entity,
                target_entity_type,
                explanation,
            ) = match

            relationship = Relationship(
                source_entity=source_entity,
                source_entity_type=source_entity_type,
                relationship=relationship,
                target_entity=target_entity,
                target_entity_type=target_entity_type,
                explanation=explanation,
            )

            relationships.append(relationship)
        return relationships

    def __call__(self, text: List[Entity]):
        self.build_prompt(text)
        response = self.llm.completion(self.prompt)

        if not response:
            raise ValueError("No response from the model")

        return self.parse_response(response)
