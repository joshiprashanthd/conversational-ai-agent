import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import download

download("wordnet")
download("omw-1.4")

lemmatizer = WordNetLemmatizer()

# Expanded keyword dictionary
keywords = {
    "stress": [
        "stress",
        "stresses",
        "stressed",
        "stressful",
        "burnout",
        "cortisol",
        "strain",
        "pressure",
        r"\bchronic stress\b",
        r"\bacute stress\b",
        "exhaustion",
        "fatigue",
        "overwhelm",
        "distress",
    ],
    "trauma": [
        "trauma",
        "traumatic",
        "PTSD",
        "post-traumatic",
        "abuse",
        "shock",
        "distress",
        "violation",
        "wound",
        "emotional injury",
        r"\btraumatic event\b",
        "flashback",
    ],
    "depression": [
        "depression",
        "depressed",
        "MDD",
        "hopelessness",
        "dysthymia",
        "anhedonia",
        "sadness",
        "melancholy",
        "low mood",
        "worthlessness",
        "major depressive disorder",
        "sorrow",
    ],
    "anxiety": [
        "anxiety",
        "anxious",
        "fear",
        "panic",
        "phobia",
        "nervousness",
        "worry",
        "stress response",
        "unease",
        "apprehension",
        "panic disorder",
        "social anxiety",
        "generalized anxiety disorder",
    ],
}


def preprocess_text(text):
    text = re.sub(r"[^\w\s]", "", text.lower())
    lemmatized_words = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(lemmatized_words)


def keyword_match(abstract, keywords, proportion=0.5):
    abstract = preprocess_text(abstract)
    results = {}

    for category, words in keywords.items():
        match_count = sum(1 for word in words if re.search(rf"\b{word}\b", abstract))
        match_proportion = match_count / len(words)
        results[category] = match_proportion >= proportion

    return results
