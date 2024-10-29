from typing import TypedDict


class Entity(TypedDict):
    entity: str
    description: str
    type: str


class Relationship(TypedDict):
    source_entity: str
    relationship: str
    target_entity: str
    explanation: str
