import json

filepath = "./extracted_entities.jsonl"

with open(filepath, "r") as f:
    data = [json.loads(line) for line in f]

for d in data:
    print("TITLE = ", d["title"])
    for rel in d["entities"]:
        types = rel["type"]
        if "Symptom" in types:
            print("\tSymptom = ", rel["entity"])
