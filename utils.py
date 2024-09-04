import re


def extract_strings(input_string, encloser):
    enclosers = {"(": ")", "[": "]", "{": "}", "<": ">"}

    if encloser not in enclosers:
        raise ValueError("Invalid encloser. Supported enclosers are (, [, {, and <")

    closing_encloser = enclosers[encloser]

    # Escape special characters for regex
    escaped_open = re.escape(encloser)
    escaped_close = re.escape(closing_encloser)

    # Find all instances of content within the specified enclosers
    pattern = f"{escaped_open}([^{escaped_open}{escaped_close}]*){escaped_close}"
    matches = re.findall(pattern, input_string)

    # Process each match
    result = []
    for match in matches:
        items = [item.strip() for item in match.split(",")]
        result.append(items)

    return result


def parse_llm_output(output):
    # Split the output into sections
    sections = re.split(r"\n\d+\.\s", output.strip())
    sections = [s.strip() for s in sections if s.strip()]

    result = {}
    for section in sections:
        # Extract the title
        title_match = re.match(r"(.+?)(?:\n|$)", section)
        if not title_match:
            continue
        title = title_match.group(1).strip()

        # Extract the content in square brackets
        square_bracket_content = re.findall(r"\[(.*?)\]", section)
        square_bracket_items = [
            item.strip()
            for item in ", ".join(square_bracket_content).split(",")
            if item.strip()
        ]

        # Extract the content in curly braces
        curly_brace_content = re.findall(r"\<(.*?)\>", section)
        curly_brace_items = [
            item.strip()
            for item in ", ".join(curly_brace_content).split(",")
            if item.strip()
        ]

        # Add to the result dictionary
        result[title] = (square_bracket_items, curly_brace_items)

    return result


def make_input(output):
    inputs = []

    INPUT_TEMP = """Parameter: {parameter}
Categories: {categories}
Points: {points}

"""

    for parameter, (categories, points) in output.items():
        inputs.append(
            INPUT_TEMP.format(
                parameter=parameter,
                categories=", ".join(categories),
                points=", ".join(points),
            )
        )

    return "".join(inputs)
