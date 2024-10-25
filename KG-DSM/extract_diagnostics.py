import argparse


filepath = "depressive-disorder.txt"

text = ""
with open(filepath, "r") as f:
    started = False
    for line in f:
        if "diagnostic criteria" in line.lower():
            started = True
        if "diagnostic features" in line.lower():
            break
        if started:
            text += line

print(text)
