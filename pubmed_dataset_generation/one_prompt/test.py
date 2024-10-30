import sys
import json

sys.path.append("../")

from Dataset import read_json_files
from OnePrompt import OnePrompt
from lib.groq_model import Llama3_1_8B_Instant, Llama3_1_70B_Versatile
from pprint import pprint

json_file_path = "../sample_abstracts.json"

publications = read_json_files([json_file_path])

model = Llama3_1_70B_Versatile(
    api_key="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"
)
prompt = OnePrompt(model)

start = 0
step = 7
end = step
response = []
while end < len(publications):
    response += prompt(publications[start:end])
    start += step
    end += step

with open("output.jsonl", "a") as f:
    for r in response:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

pprint(response)
