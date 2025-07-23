import re

import contractions


def preprocessing(text):
    # 1. Expand contractions
    expanded_text = contractions.fix(text)

    # 2. Convert to lowercase
    expanded_text = expanded_text.lower()

    # 3. Remove punctuation, digits, etc.
    # (Example: keep only letters and spaces)
    cleaned_text = re.sub(r"[^a-z\s]", "", expanded_text)

    # 4. Tokenize, remove stopwords, etc.
    # ... (rest of your preprocessing) ...

    return cleaned_text


# --- Example Usage ---
text1 = "He can't go, it's not safe."
text2 = "They're running late, shouldn't we wait?"

print(f"Original: {text1}")
print(f"Processed: {preprocessing(text1)}")
print("-" * 20)
print(f"Original: {text2}")
print(f"Processed: {preprocessing(text2)}")
