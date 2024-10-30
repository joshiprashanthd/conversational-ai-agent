import pandas as pd
import json
import os


def generate_json(file_name):
    path = f"./{file_name}"
    type = os.path.splitext(file_name)[0].split("-")[1].strip()
    data = pd.read_csv(path).astype(str)
    output = []

    for _, row in data.iterrows():
        entry = {
            "type": type,
            "description": row["name"].strip(),
            "reasoning": row["description how it solves"].strip(),
            "howto": row["description how to perform"].strip(),
            "category": row["category"].split(", "),
            "effectiveness": row["effectiveness"].strip(),
        }
        output.append(entry)

    output_json = json.dumps(output, indent=4)
    output_path = f"./{type}.json"

    with open(output_path, "w") as f:
        f.write(output_json)


csv_files = [file for file in os.listdir(".") if os.path.splitext(file)[1] == ".csv"]

for csv_file in csv_files:
    generate_json(csv_file)
