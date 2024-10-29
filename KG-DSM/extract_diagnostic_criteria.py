import json

from pathlib import Path

text_filepath = "depressive-disorder.txt"


target_dir_path = Path("./diagnostic_criterias")
target_dir_path.mkdir(parents=True, exist_ok=True)

res = {}
with open(text_filepath, "r") as f:
    prev_line = ""
    title = ""
    text = ""

    for line in f:
        if "diagnostic criteria" in line.lower():
            title = prev_line
        if "diagnostic features" in line.lower():
            if len(title) > 0:
                res[title.strip()] = text.strip()
            text = ""
            title = ""

        if len(title) > 0:
            text += line

        prev_line = line

filepath = target_dir_path / (Path(text_filepath).stem + ".jsonl")

with open(filepath, "w") as f:
    for title, text in res.items():
        f.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")
