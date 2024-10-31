import sys
import json

sys.path.append("../")

from ExtractEntities import ExtractEntities
from lib.groq_model import Llama3_1_8B_Instant
from pprint import pprint
from tqdm import tqdm

json_file_path = "../../full_diagnostic_criterias/depressive-disorder.jsonl"
target_json_path = "./extracted_entities.jsonl"

data = []
with open(json_file_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

model = Llama3_1_8B_Instant(
    api_key="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"
)

prompt = ExtractEntities(model)

target_keys = [
    "diagnostic_features",
    "comorbidity",
    "functional_consequences",
    "risk_and_prognostic_factors",
]


with open(target_json_path, "w") as f:
    for d in tqdm(data):
        res = {"title": d["title"]}

        text = ""
        for key, value in d["text"].items():
            text += (
                "### " + key + " ###" + "\n\n"
            )  # add title of the section as well to the text

            if any(k in key for k in target_keys):
                text += value
            text += value

        response = prompt(text)
        res["entities"] = response

        f.write(json.dumps(res, ensure_ascii=False) + "\n")
