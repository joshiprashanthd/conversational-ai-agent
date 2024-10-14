import json
import os

from groq_model import (
    Llama3_1_8B_Instant,
    Llama3_1_70B_Versatile,
    Llama3_1_405B_Reasoning,
)

model1 = Llama3_1_70B_Versatile()
model2 = Llama3_1_8B_Instant()

def prompt(response, category):
    prompt = f"""
    Role: Assume the role of a human annotator/psychologist/therapist  tasked with analyzing user responses. Your goal is to identify and extract patterns based on the nature of the user’s tendencies or the underlying faculty driving their behavior.

    Task Objective: 
    Your job is to extract either Nature or Faculty nodes (but not both) from a given user response. The type of node you will extract is specified by the provided category label.

    Nature Nodes: 
    Identify and extract tendencies, preferences, or behaviors that reflect the user's lifestyle, habits, or emotional inclinations. Ensure that the node captures the user's behavior concisely, while preserving the context of the entire response.

    Example of Nature Nodes:
    Enjoys peaceful life
    Likes eating homemade food
    Prefers calm, solitary activities

    Faculty Nodes: 
    Extract cognitive or emotional factors such as the user’s thought patterns, memory, willpower, or decision-making tendencies. Ensure that these nodes reflect underlying internal factors that shape their behavior.

    Example of Faculty Nodes:
    Struggles with confidence
    Overthinks past mistakes
    Avoids risky decisions due to fear of failure

    Important Note:
    While extracting nodes, do not lose the context of the entire response. A piece of text might seem to represent a node in isolation, but its true meaning could be influenced by surrounding sentences. Your task is to ensure that the meaning of each node remains faithful to the user’s full response.

    Instructions:
    - Focus on the category (either Nature or Faculty) specified in the prompt.
    - Maintain the context of the entire response while identifying the patterns.
    - Ensure each node is concise but contextually accurate to the user’s response.
    - Extract multiple relevant nodes from the user response, with each node separated by a new line.
    - Do not start with irrelevant phrases like "extracted nature nodes:" give the nodes directly
    - for each faculty also give the category of the faculty node from these three (intelligence, willpower, memory)

    Example:
    1. For Nature:
        User response:
        "I feel really happy when I’m outside, especially near a park or any place with trees. I also love listening to calming music, especially ones that help me relax. Eating a good meal, especially homemade food, gives me a lot of satisfaction. Whenever I have free time, I either like to stay home and binge-watch TV shows or read a book."

        Extracted Nature Nodes:
        Enjoys spending time in nature
        Finds peace in listening to calming music
        Derives satisfaction from eating homemade food
        Prefers relaxing at home

    2. For Faculty:
        User response:
        "I struggle to stay organized, especially when there are a lot of things going on at once. I start with good intentions, but then I lose track of what needs to be done and get overwhelmed. I’ve tried making to-do lists, but I often forget to check them. I usually second-guess myself a lot when I have to make a big decision. Even if I’ve researched everything, I still worry about making the wrong choice."

        Extracted Faculty Nodes:
        Struggles with organization due to overwhelm
        Lack of consistent planning or follow-through
        Second-guesses decisions, even after researching
        Worries about making the wrong choice

    Now extract the relevant nodes from the following response:
    Category: {category}
    User Response: {response}
    ""
    """

    return prompt




NatureResponse = """
I feel really happy when I’m outside, especially near a park or any place with trees. I also love listening to calming music, especially ones that help me relax. Eating a good meal, especially homemade food, gives me a lot of satisfaction. Whenever I have free time, I either like to stay home and binge-watch TV shows or read a book.
"""

FacultyResponse = """
 struggle to stay organized, especially when there are a lot of things going on at once. I start with good intentions, but then I lose track of what needs to be done and get overwhelmed. I’ve tried making to-do lists, but I often forget to check them. I usually second-guess myself a lot when I have to make a big decision. Even if I’ve researched everything, I still worry about making the wrong choice.
"""

nature_result = model1.completion(prompt(NatureResponse, 'Nature'))
faculty_result = model1.completion(prompt(FacultyResponse, 'Faculty'))

nature_nodes = nature_result.strip().split('\n')
faculty_nodes = faculty_result.strip().split('\n')
nodes_dict = {
    'Nature': nature_nodes,
    'Faculty': faculty_nodes
}

# JSON file path
# make file path same as the file where current file is stored
json_file_path = os.path.join(os.path.dirname(__file__), 'nodes.json')

# Function to save nodes to JSON file
def save_to_json(nodes_dict, file_path):
    if os.path.exists(file_path):
        # Load existing data if the file exists
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    else:
        # Start with an empty dict if the file doesn't exist
        existing_data = {'Nature': [], 'Faculty': []}

    # Append new nodes to the existing data
    existing_data['Nature'].extend(nodes_dict['Nature'])
    existing_data['Faculty'].extend(nodes_dict['Faculty'])

    # Remove duplicates
    existing_data['Nature'] = list(set(existing_data['Nature']))
    existing_data['Faculty'] = list(set(existing_data['Faculty']))

    # Save back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)

    print(f"Nodes saved to {file_path}")


# Call the function to save extracted nodes
save_to_json(nodes_dict, json_file_path)