from groq_model import (
    Llama3_1_8B_Instant,
    Llama3_1_70B_Versatile,
    Llama3_1_405B_Reasoning,
)

from utils import parse_llm_output, make_input

model = Llama3_1_8B_Instant()
# model = Llama3_1_70B_Versatile()

prompt = """You are an expert mental health professional. Your goal is to read patient's answers to some of the questions given and find out the category (or categories) they belong to and extract relevant phrases from the answer which can be useful to assess patient's mental state.

Task Instructions:
	- You can pick multiple categories which are most suitable for the answer.
	- Categories for each question are provided, do not create categories of your own.
    - All ways add number and the title of the question before generating categories.
    - Do not repeat answer in your response.
	- Enclose all the comma separated categories you generate with left bracket [ and right bracket ]
	- Enclose each the comma separated extracted phrases in left parentheses ( and right parentheses ).
	

1. Noise during sleep 
Question: Can you tell me a little about the noise level in your sleep environment? Is it usually quiet, or are there disturbances like traffic or household noises?
Categories: [quiet, moderate noise, very noise]
Answer: I don't really pay attention to noise. It's always there. There's this constant hum in my head, so other noises don't really bother me.

2. Temperature During Sleep
Question: How would you describe the temperature in your bedroom when you sleep? Is it too warm, too cold, or just right?
Categories: [too cold, comfortable, too hot]
Answer: I don't know. I'm usually cold, even when the heat is on. Nothing seems to warm me up.

3. Sleep Continuity
Question: Do you tend to sleep through the night without waking up, or do you find yourself waking up several times? If so, can you tell me a bit about that?
Categories: [Undisturbed, Few interruptions 1 to 2 times, Frequent interruptions 3+ times, Very fragmented sleep]
Answer: I wake up all the time. I don't know why. It's like my brain won't shut off. I just lie there staring at the ceiling.

4. Thoughts Before Sleep
Question: What kind of thoughts do you usually have when you’re trying to fall asleep? Are they calm and relaxing, or do you find yourself worrying about things?
Categories: [excited, hopeful, anxious, worried, blank, forgetful, planning, strategizing]
Answer: My mind just races. I think about everything that went wrong that day, and then I start worrying about tomorrow. It's a never-ending cycle.

5. Thoughts After Waking Up
Question: How do you typically feel when you wake up in the morning? Refreshed and ready to start the day, or tired and groggy?
Categories: [Refreshed, Tired, Hurried, Concerned]
Answer: I feel like a zombie. I don't want to get out of bed. Everything feels heavy and pointless.

6. Kind of Dreams
Question: Can you tell me a bit about your dreams? Do you remember them often? Are they usually pleasant, or do you have nightmares or disturbing dreams?
Categories: [Pleasant, Neutral, Nightmares, Vivid, Lucid, Recurring]
Answer: I don't remember my dreams. Or maybe I do, but they're just a blur. Nothing important.


Extract relevant phrases from the answer and assign categories for each answer:
"""

response = model.completion(prompt)
output = parse_llm_output(response)


prompt = """You are an expert mental health professional. You are given patient's input to various questions related to various parameters such as "Noise during sleep". Below are the rules for each parameter that need to follow to reach to a detailed conclusion. Conclusion should only be one paragraph.

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

input = make_input(output)


output = model.completion(prompt.format(input=input))

print(output)
