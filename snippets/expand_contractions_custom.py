import re

CONTRACTION_MAP = {
    "can't": "cannot",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    # ... add more as needed
}


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    # Create a regex pattern from the dictionary keys
    contractions_pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL,
    )

    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contraction_mapping.get(match.lower())
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text


# --- Example Usage ---
text = "He can't go, it's not safe."
print(f"Original: {text}")
print(f"Expanded: {expand_contractions(text)}")
