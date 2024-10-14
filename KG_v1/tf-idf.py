from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# Load nodes from a JSON file
json_file_path = os.path.join(os.path.dirname(__file__), 'nodes.json')
with open(json_file_path, 'r') as file:
    nodes_dict = json.load(file)

# Separate nature and faculty nodes
nature_nodes = nodes_dict['Nature']
faculty_nodes = nodes_dict['Faculty']

# Step 1: Combine Nature and Faculty nodes into a single list for vectorization
all_nodes = nature_nodes + faculty_nodes

# Step 2: Convert the nodes into TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_nodes)

# Step 3: Calculate Cosine Similarity between Nature (first len(nature_nodes)) and Faculty nodes
similarity_matrix = cosine_similarity(tfidf_matrix[:len(nature_nodes)], tfidf_matrix[len(nature_nodes):])

# Step 4: Create a dictionary to store the links between Nature nodes and Faculty nodes
linked_nodes = []

for i, nature_node in enumerate(nature_nodes):
    # Create an entry for each Nature node
    nature_entry = {
        "Nature Node": nature_node,
        "Links": []
    }

    # Display each Faculty node and its similarity to the current Nature node
    for j, faculty_node in enumerate(faculty_nodes):
        similarity = similarity_matrix[i, j]
        if similarity > 0.1:  # Adjust this threshold as needed to filter weak links
            nature_entry["Links"].append({
                "Faculty Node": faculty_node,
                "Similarity": round(similarity, 2)  # Rounding for cleaner display
            })
    
    # Add the nature entry to the linked_nodes list
    linked_nodes.append(nature_entry)

# Step 5: Append the linked nodes to an existing JSON file

# Path to the output JSON file
output_file_path = os.path.join(os.path.dirname(__file__), 'linked_nodes_tf-idf.json')

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
