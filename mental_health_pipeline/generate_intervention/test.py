import sys

sys.path.append("../..")
from lib.groq_model import Llama3_1_70B_Versatile
from GenerateIntervention import GenerateIntervention, Input

model = Llama3_1_70B_Versatile()
prompt = GenerateIntervention(model)

input: Input = {
    "parameter": "Ease of falling asleep",
    "current_state": "difficult",
    "target_state": "easy",
    "topic": "Meditations",
}

print(prompt(input))
