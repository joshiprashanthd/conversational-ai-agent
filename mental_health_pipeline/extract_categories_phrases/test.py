import sys

sys.path.append("../..")

from lib.groq_model import Llama3_1_70B_Versatile
from typing import List
from ExtractCategoriesPhrases import ExtractCategoriesPhrases, QNA
from pprint import pprint

sleep_data: List[QNA] = [
    {
        "parameter": "Noise during sleep",
        "question": "Can you tell me a little about the noise level in your sleep environment? Is it usually quiet, or are there disturbances like traffic or household noises?",
        "categories": ["quiet", "moderate noise", "very noise"],
        "answer": "I don't really pay attention to noise. It's always there. There's this constant hum in my head, so other noises don't really bother me.",
    },
    {
        "parameter": "Temperature During Sleep",
        "question": "How would you describe the temperature in your bedroom when you sleep? Is it too warm, too cold, or just right?",
        "categories": ["too cold", "comfortable", "too hot"],
        "answer": "I don't know. I'm usually cold, even when the heat is on. Nothing seems to warm me up.",
    },
    {
        "parameter": "Sleep Continuity",
        "question": "Do you tend to sleep through the night without waking up, or do you find yourself waking up several times? If so, can you tell me a bit about that?",
        "categories": [
            "Undisturbed",
            "Few interruptions 1 to 2 times",
            "Frequent interruptions 3+ times",
            "Very fragmented sleep",
        ],
        "answer": "I wake up all the time. I don't know why. It's like my brain won't shut off. I just lie there staring at the ceiling.",
    },
    {
        "parameter": "Thoughts Before Sleep",
        "question": "What kind of thoughts do you usually have when you’re trying to fall asleep? Are they calm and relaxing, or do you find yourself worrying about things?",
        "categories": [
            "excited",
            "hopeful",
            "anxious",
            "worried",
            "blank",
            "forgetful",
            "planning",
            "strategizing",
        ],
        "answer": "My mind just races. I think about everything that went wrong that day, and then I start worrying about tomorrow. It's a never-ending cycle.",
    },
    {
        "parameter": "Thoughts After Waking Up",
        "question": "How do you typically feel when you wake up in the morning? Refreshed and ready to start the day, or tired and groggy?",
        "categories": ["Refreshed", "Tired", "Hurried", "Concerned"],
        "answer": "I feel like a zombie. I don't want to get out of bed. Everything feels heavy and pointless.",
    },
    {
        "parameter": "Kind of Dreams",
        "question": "Can you tell me a bit about your dreams? Do you remember them often? Are they usually pleasant, or do you have nightmares or disturbing dreams?",
        "categories": [
            "Pleasant",
            "Neutral",
            "Nightmares",
            "Vivid",
            "Lucid",
            "Recurring",
        ],
        "answer": "I don't remember my dreams. Or maybe I do, but they're just a blur. Nothing important.",
    },
]

model = Llama3_1_70B_Versatile()
prompt = ExtractCategoriesPhrases(model)

pprint(prompt(sleep_data))
