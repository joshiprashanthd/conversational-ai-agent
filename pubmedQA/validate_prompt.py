#Prompt 1: Validating topic for Question Generation
#Objective: Check if the given topic is related to mental health topics

import json

def validate_topic(subject, topic):
    prompt = f"""
    You are a mental health professional evaluating the relevance of a subject and its associated topic. Your task is to assess whether the provided subject and topic are relevant to mental health. The answer should be either Yes or No.

    **Instructions**:
    1. If the subject and topic are directly related to mental health, mark it **Yes**.
    2. If the subject and topic are unrelated to mental health, mark it **No**.
    3. Provide a brief explanation justifying your answer, mentioning how the subject and topic connect (or do not connect) to mental health.

    **Input Example**:
    Subject: "Psychology"
    Topic: "The Impact of Social Media on Adolescent Mental Health"

    **Output Format**:
    - **Answer**: [Yes or No]
    - **Justification**: [Provide a brief explanation justifying your decision.]

    **Expected Output Example**:
    - **Answer**: Yes
    - **Justification**: The subject is "Psychology," which directly pertains to mental health, and the topic focuses on how social media influences adolescent mental well-being, making it relevant to mental health.

    **Now evaluate the provided Subject and Topic to determine if they are relevant to mental health**:
    Subject: {subject}
    Topic: {topic}
    """

    return prompt.format(subject=subject, topic=topic)
