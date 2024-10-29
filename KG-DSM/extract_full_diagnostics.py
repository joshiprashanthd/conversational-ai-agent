import json
from pathlib import Path
import re

criteria_list = [
    "Diagnostic Criteria",
    "Recording Procedures",
    "Diagnostic Features",
    "Associated Features",
    "Prevalence",
    "Development and Course",
    "Risk and Prognostic Factors",
    "Culture-Related Diagnostic Issues",
    "Diagnostic Markers",
    "Association with",
    "Sex- and Gender-Related Diagnostic Issues",
    "Functional Consequences of",
    "Differential Diagnosis",
    "Comorbidity",
]


def extract_diagnostics(text_filepath, target_dir_path):

    target_dir_path = Path(target_dir_path)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    sluggify = lambda x: re.sub(r"[^\w]+", "_", x.strip()).lower()

    res = {}
    with open(text_filepath, "r") as f:
        prev_line = ""
        title = ""
        subtitle = ""
        text = ""

        line_num = 0
        lines = f.readlines()
        while line_num < len(lines):
            line = lines[line_num]

            if "diagnostic criteria" in line.lower() and len(line.split(" ")) < 7:
                if len(subtitle) > 0:
                    res[title][subtitle] = text

                title = sluggify(prev_line)
                res[title] = {}
                text = ""
                subtitle = ""

            if (
                any([criteria.lower() in line.lower() for criteria in criteria_list])
                and len(line.split()) < 7
            ):
                if len(subtitle) > 0:
                    res[title][subtitle] = text
                    text = ""

                subtitle = sluggify(line)

            text += re.sub(r"\s+", " ", line).strip() + "\n"
            prev_line = line

            line_num += 1

    filepath = target_dir_path / (Path(text_filepath).stem + ".jsonl")

    with open(filepath, "w") as f:
        for title, text in res.items():
            f.write(
                json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n"
            )


extract_diagnostics("depressive-disorder.txt", "./full_diagnostic_criterias")
