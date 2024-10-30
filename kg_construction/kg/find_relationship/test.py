import sys
import json

sys.path.append("../")

from FindRelationship import FindRelationship
from lib.groq_model import Llama3_1_8B_Instant, Llama3_1_70B_Versatile
from pprint import pprint
from tqdm import tqdm

json_file_path = "../extract_entities/extracted_entities.jsonl"
target_json_path = "./extracted_relationships.jsonl"

data = []
with open(json_file_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

model = Llama3_1_70B_Versatile(
    api_key="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"
)

prompt = FindRelationship(model)


with open(target_json_path, "w") as f:
    for d in tqdm(data):
        res = {"title": d["title"]}
        entities = d["entities"]

        response = prompt(entities)
        res["relationships"] = response

        f.write(json.dumps(res, ensure_ascii=False) + "\n")
