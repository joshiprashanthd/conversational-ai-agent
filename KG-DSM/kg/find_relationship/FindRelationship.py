# import sys, re

# sys.path.append("../")

import re
from typing import List
from groq_model import GroqModel
from kg.types import Entity, Relationship


class FindRelationship:
    def __init__(self, llm: GroqModel):
        self.llm = llm

    def build_prompt(self, entities: List[Entity]):
        self.prompt = """You are an expert in the field of knowledge graph and mental health. Your goal is to find the relationships between entities.

Task Instructions:
    - You will be given entity name, description, and type of it.
    - Use description to find the relationships between entities.
    - You can use the following relationships: causes, treats, prevents, has_metric, assessed_in ,  

Output Format:
Source Entity: <name of the source entity>
Relationship: <relationship between source and target entity>
Target Entity: <name of the target entity>
Explanation: <explaination of relationship between source and target entity>

Here are the entities:
{entities}
"""
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
            r"Source Entity: (.+?)\nRelationship: (.+?)\nTarget Entity: (.+?)\nExplanation: (.+?)\n"
        )
        matches = pattern.findall(response)
        relationships = []
        for match in matches:
            source_entity, relationship, target_entity, explanation = match
            relationship = Relationship(
                source_entity=source_entity,
                relationship=relationship,
                target_entity=target_entity,
                explanation=explanation,
            )
            relationships.append(relationship)
        return relationships

    def __call__(self, text: str):
        self.build_prompt(text)
        response = self.llm.completion(self.prompt)
        return self.parse_response(response)
