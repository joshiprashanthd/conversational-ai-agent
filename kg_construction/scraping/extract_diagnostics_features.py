import json
from pathlib import Path

text_filepath = "depressive-disorder.txt"

target_dir_path = Path("./functional_consequences")
target_dir_path.mkdir(parents=True, exist_ok=True)

res = {}
with open(text_filepath, "r") as f:
    title = ""
    text = ""

    for line in f:
        if len(title) == 0 and "diagnostic features" in line.lower():
            if " of " in line.lower():
                title = line.split(" of ")[1].strip()
            else:
                title = line.strip()

        if "differential diagnosis" in line.lower():
            if len(title) > 0:
                res[title] = text.strip()
            text = ""
            title = ""

        if len(title) > 0:
            text += line

filepath = target_dir_path / (Path(text_filepath).stem + ".jsonl")
with open(filepath, "w") as f:
    for title, text in res.items():
        f.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")
