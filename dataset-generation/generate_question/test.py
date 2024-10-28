import sys

sys.path.append("../")

from Dataset import read_json_files
from GenerateQuestions import GenerateQuestions
from groq_model import Llama3_1_70B_Versatile
from pprint import pprint

json_file_path = "../sample_abstracts.json"

publications = read_json_files([json_file_path])[:1]

model = Llama3_1_70B_Versatile(
    api_key="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"
)

prompt = GenerateQuestions(model)

responses = [prompt(pub) for pub in publications]
pprint(responses)
