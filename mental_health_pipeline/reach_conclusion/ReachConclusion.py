from lib.groq_model import GroqModel
from typing import TypedDict, List


class DictUnit(TypedDict):
    parameter: str
    states: List[str]
    phrases: List[str]


TReachConclusionInput = List[DictUnit]


class ReachConclusion:
    def __init__(self, llm: GroqModel):
        self.llm = llm

    def build_prompt(self, input: TReachConclusionInput):
        self.prompt = """You are an expert mental health professional. You are given patient's input to various questions related to various parameters such as "Noise during sleep". Below are the rules for each parameter that need to follow to reach to a detailed conclusion. Conclusion should only be one paragraph.

Input Description:
    - Parameter: name of the parameter.
    - Categories: defines a simple explanation of what patient is trying to tell us about the parameter.
    - Points: relevant phrases extracted from the patient response. These phrases can provide relevant information about the parameter.

Rules:
1. Noise During Sleep
Criteria: Frequency, intensity, type of noise, impact on sleep quality.
Analysis: Evaluate the frequency and intensity of noise to determine potential sleep disruptions. Consider the type of noise (e.g., traffic, household) as it might indicate specific environmental stressors.
Conclusion: Determine if noise is a significant factor affecting sleep quality and overall well-being.

2. Temperature During Sleep
Criteria: Preferred temperature, temperature fluctuations, impact on sleep comfort.
Analysis: Assess the patient's ideal sleep temperature and any discrepancies between preferred and actual temperature. Consider the frequency of temperature fluctuations and their impact on sleep quality.
Conclusion: Determine if temperature is a contributing factor to sleep disturbances or overall discomfort.

3. Sleep Continuity
Criteria: Frequency of awakenings, duration of awakenings, reasons for awakenings.
Analysis: Evaluate the number and duration of awakenings per night. Consider the patient's reported reasons for waking up (e.g., bathroom, discomfort, worries).
Conclusion: Determine the severity of sleep fragmentation and potential underlying causes.

4. Thoughts Before Sleep
Criteria: Content of thoughts (positive, negative, neutral), intensity of thoughts, frequency of occurrence.
Analysis: Assess the nature of the patient's pre-sleep thoughts. Consider the emotional tone and how these thoughts impact sleep onset.
Conclusion: Determine if pre-sleep thoughts are contributing to sleep difficulties or reflecting underlying emotional issues.

5. Thoughts After Waking Up
Criteria: Mood upon waking, energy levels, cognitive function.
Analysis: Evaluate the patient's emotional state and energy levels immediately after waking. Consider any difficulties with concentration or focus.
Conclusion: Determine the impact of sleep quality on overall mood and daytime functioning.

6. Kind of Dreams
Criteria: Dream content (positive, negative, neutral), frequency of nightmares, vividness of dreams.
Analysis: Assess the emotional tone of the dreams and their potential impact on sleep quality. Consider the frequency of nightmares as an indicator of potential distress.
Conclusion: Determine if dream patterns are related to sleep disturbances or emotional well-being.

Read categories and points carefully and follow the rules to reach to a conclusion.

Input:
{input}

Conclusion:
"""

        inputs = []

        INPUT_TEMP = """Parameter: {parameter}
    Categories: {states}
    Points: {phrases}
    """

        for d in input:
            inputs.append(
                INPUT_TEMP.format(
                    parameter=d["parameter"],
                    states=", ".join(d["states"]),
                    phrases=", ".join(d["phrases"]),
                )
            )

        input_text = "".join(inputs)
        self.prompt = self.prompt.format(input=input_text)

    def process_response(self, output: str):
        return output

    def __call__(self, input: TReachConclusionInput):
        self.build_prompt(input)
        response = self.llm.completion(self.prompt)
        return self.process_response(response)
