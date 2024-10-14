from sentence_transformers import SentenceTransformer, util
import json
import os

# Load a pre-trained BERT model for generating sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the nature nodes and faculty nodes from the JSON file
json_file_path = os.path.join(os.path.dirname(__file__), 'nodes.json')
with open(json_file_path, 'r') as file:
    nodes_dict = json.load(file)

nature_nodes = nodes_dict['Nature']
faculty_nodes = nodes_dict['Faculty']

# Step 1: Generate embeddings for both Nature and Faculty nodes
nature_embeddings = model.encode(nature_nodes, convert_to_tensor=True)
faculty_embeddings = model.encode(faculty_nodes, convert_to_tensor=True)

# Step 2: Calculate cosine similarity between Nature and Faculty embeddings
cosine_scores = util.pytorch_cos_sim(nature_embeddings, faculty_embeddings)

# Step 3: Create a dictionary to store the links between Nature nodes and Faculty nodes
linked_nodes = []

for i, nature_node in enumerate(nature_nodes):
    # Create an entry for each Nature node
    nature_entry = {
        "Nature Node": nature_node,
        "Links": []
    }

    # Iterate over each Faculty node and its similarity to the current Nature node
    for j, faculty_node in enumerate(faculty_nodes):
        similarity = cosine_scores[i][j].item()  # Extract similarity value from tensor
        if similarity > 0.1:  # Adjust the threshold to filter weak links if necessary
            nature_entry["Links"].append({
                "Faculty Node": faculty_node,
                "Similarity": round(similarity, 2)  # Rounded for better readability
            })
    
    # Append the nature entry to the linked_nodes list
    linked_nodes.append(nature_entry)

# Step 4: Append the new linked nodes to an existing JSON file

# Path to the output JSON file
output_file_path = os.path.join(os.path.dirname(__file__), 'linked_nodes_bert.json')

# Check if the output file already exists
if os.path.exists(output_file_path):
    # Load the existing data from the JSON file
    with open(output_file_path, 'r') as output_file:
        existing_data = json.load(output_file)
else:
    existing_data = []

# Append new linked nodes to the existing data
existing_data.extend(linked_nodes)

# Save the updated data back to the JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(existing_data, output_file, indent=4)

print(f"Links between Nature and Faculty nodes appended to {output_file_path}")
