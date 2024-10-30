from dataclasses import dataclass
import json


@dataclass
class Publication:
    pmid: str
    title: str
    abstract: str


def read_json_files(file_paths):
    publications = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data = json.load(f)
            for item in data:
                publication = Publication(**item)
                publications.append(publication)
    return publications
