import sys

sys.path.append("../")

from Dataset import read_json_files
from MentalHealthCategory import MentalHealthCategory
from groq_model import Llama3_1_70B_Versatile
from pprint import pprint

json_file_path = "../sample_abstracts.json"

publications = read_json_files([json_file_path])

model = Llama3_1_70B_Versatile(
    api_key="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"
)
prompt = MentalHealthCategory(model)

response = prompt(publications)
pprint(response)
