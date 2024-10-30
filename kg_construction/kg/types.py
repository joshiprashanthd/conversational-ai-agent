from typing import TypedDict


class Entity(TypedDict):
    entity: str
    description: str
    type: str


class Relationship(TypedDict):
    source_entity: str
    source_entity_type: str
    relationship: str
    target_entity: str
    target_entity_type: str
    explanation: str
