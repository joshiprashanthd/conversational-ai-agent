#Prompt 1: Validating Abstract for Question Generation
#Objective: Check if the given abstract is suitable for generating a meaningful question.

def validate_abstract(title, abstract):
    prompt = f"""
    You are a biomedical text evaluator. Your task is to assess whether the provided title and abstract contain enough concrete, interpretable scientific findings to generate a useful question-answer pair. The answer should be either Yes or No.

    Instructions:
    1. If the abstract contains specific research outcomes, results, or evidence, mark it Yes.
    2. If the abstract is too general, purely theoretical, a review, or lacks interpretable findings, mark it No.
    3. Provide a short justification explaining your answer, in 1-2 sentences.

    Input Example:

    Title: "The Role of Vitamin D in Reducing Inflammation in Elderly Patients"
    Abstract:
    "Recent studies have shown a significant correlation between low vitamin D levels and increased inflammatory markers in elderly individuals. In a double-blind clinical trial, elderly patients supplemented with 2000 IU of vitamin D daily showed a 30% reduction in C-reactive protein levels over six months, suggesting the potential of vitamin D in managing chronic inflammation."

    Expected Output Example:
    Answer: Yes
    Justification: The abstract contains concrete findings from a clinical trial with measurable outcomes that can generate a meaningful question-answer pair.

    Now evaluate the title and abstract to determine if it is suitable for generating a question:
    Title: {title}
    Abstract: {abstract}

    """

    return prompt.format(title=title, abstract=abstract)
