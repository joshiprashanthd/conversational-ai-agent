# Prompt 2: Generating Questions and Distractors from the Abstract
# Objective: Create a question-answer pair from the abstract and generate three distractors (incorrect but related answers).


def prompt_QA(title, abstract):

    prompt = f"""
    You are a biomedical question generator. Based on the provided title and abstract, generate a fact-based multiple-choice question (MCQ). The question should relate to the title, while the answer and distractors should be based on findings from the abstract. Ensure that all answer options are short and clear, and that the generated question can stand alone without needing to refer back to the abstract.

    **Instructions**:
    1. **Question**: Create a factual multiple-choice question (MCQ) that is based on or inspired by the title. Ensure the question is aligned with the research topic indicated by the title.
    2. **Answer Options (optA, optB, optC, optD)**: Provide four short, clear, and distinct answer options:
        - One correct answer based on the abstract’s findings.
        - Three incorrect but plausible distractors derived from the abstract or general knowledge about the topic.
    3. **Correct Answer**: Identify the correct answer from the options (optA, optB, optC, or optD).
    4. **Explanation**: Provide a brief explanation that justifies why the correct answer is correct, and explain why the distractors are incorrect.
    5. **Subject**: Mention the broad subject area (e.g., Biochemistry, Surgery, Anatomy).
    6. **Topic**: Specify the topic within the subject (e.g., Vitamin Deficiency, Surgical Procedures).

    Strictily follow the output format provided below. If the abstract does not contain sufficient information to generate a meaningful question, leave the question field empty.
    **Output Format**:
    - Question: [The question text inspired by the title]
    - optA: [First option]
    - optB: [Second option]
    - optC: [Third option]
    - optD: [Fourth option]
    - Correct Answer: [e.g., optA]
    - Explanation: [A brief explanation justifying the correct answer and explaining why the distractors are incorrect.]
    - Subject: [e.g., Psychology]
    - Topic: [e.g., Mental Health]

    **Input Example**:
    Title: "The Role of Vitamin D in Reducing Inflammation in Elderly Patients"
    Abstract: 
    "Recent studies show a significant correlation between low vitamin D levels and increased inflammatory markers in elderly individuals. In a clinical trial, patients supplemented with 2000 IU of vitamin D daily showed a 30% reduction in C-reactive protein levels."

    **Expected Output Example**:
    - Question: How does Vitamin D supplementation affect inflammation in elderly patients?
    - optA: It reduces inflammatory markers by 30%
    - optB: It has no effect on inflammation
    - optC: It increases inflammation levels
    - optD: It only reduces cholesterol levels
    - Correct Answer: optA
    - Explanation: The abstract shows that vitamin D supplementation reduced inflammatory markers by 30%. The other options either deny this or mention unrelated effects.
    - Subject: Biochemistry
    - Topic: Vitamin D and Inflammation

    **Now generate a question-answer pair based on the following title and abstract**:
    Title: {title}
    Abstract: {abstract}
    """

    return prompt.format(title=title, abstract=abstract)
