import json

filepath = "./depressive-disorder.jsonl"

with open(filepath, "r") as f:
    data = [json.loads(line) for line in f]


for d in data:
    print("TITLE = ", d["title"])
    for key, value in d["text"].items():
        # print("\tDiagnostic Features = ", value)
        print("\tKey = ", key)
    print("\n\n\n")
