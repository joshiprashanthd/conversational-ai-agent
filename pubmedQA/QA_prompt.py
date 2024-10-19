#Prompt 2: Generating Questions and Distractors from the Abstract
#Objective: Create a question-answer pair from the abstract and generate three distractors (incorrect but related answers).

def prompt_QA(title, abstract):

    prompt = f""" 
    You are a biomedical question generator. Your role is to craft a well-formed multiple-choice question (MCQ) from the provided abstract, ensuring the correct answer is based on the scientific findings in the text. Additionally, generate three incorrect but plausible distractor options.

    Instructions:
    Question: Formulate a concise yes/no, fact-based, or knowledge-based question aligned with the abstract’s findings.
    Correct Answer: The correct answer should directly reflect the abstract’s key findings or results.
    Distractors: Create three plausible incorrect answers that are close to the findings but factually incorrect.
    Ensure that all answer choices are in similar formats (e.g., all answers as Yes/No or numerical values if applicable).
    Avoid overly obvious distractors; they should require some reasoning to distinguish from the correct answer.

    Input Example:
    Title: "The Role of Vitamin D in Reducing Inflammation in Elderly Patients"
    Abstract:
    "Recent studies have shown a significant correlation between low vitamin D levels and increased inflammatory markers in elderly individuals. In a double-blind clinical trial, elderly patients supplemented with 2000 IU of vitamin D daily showed a 30% reduction in C-reactive protein levels over six months, suggesting the potential of vitamin D in managing chronic inflammation."

    Expected Output Example:
    Question: Does supplementation with 2000 IU of vitamin D daily reduce inflammatory markers in elderly patients?
    Correct Answer: Yes
    Distractor 1: No
    Distractor 2: Only a reduction in blood pressure was observed.
    Distractor 3: The study found no difference in inflammation levels after supplementation.

    stick to the output format to generate a question-answer pair and distractors based on the provided abstract:

    Now generate a question-answer pair and distractors based on the provided abstract:
    Title: {title}
    Abstract: {abstract}
    """

    return prompt