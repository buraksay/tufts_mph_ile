import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- 1. Setup and Sample Data ---
# Ensure you have the necessary NLTK data
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")

# Create a sample DataFrame (replace this with your actual data loading)
data = {
    "narrative": [
        "Patient reports severe chest pain and shortness of breath. History of hypertension.",
        "Follow-up for patient's chronic back pain. No new issues reported.",
        "Routine check-up. Patient is doing well. All vitals are stable.",
        "Called 911 for acute shortness of breath. Possible allergic reaction.",
        "Minor cut on the left hand. Cleaned and bandaged. Tetanus shot given.",
        "The patient has a history of back issues and reports pain after a fall.",
        "This is just a test entry to see how the word 'test' is handled.",
    ]
}
df = pd.DataFrame(data)


# --- 2. Preprocessing Function ---
def preprocess_text(text):
    """Cleans and tokenizes a single text string."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


# Apply the preprocessing to your narratives
df["tokens"] = df["narrative"].apply(preprocess_text)

# --- 3. Calculate Word Frequencies ---
# Create a single list of all words from all narratives
all_words = [word for tokens_list in df["tokens"] for word in tokens_list]

# Count the frequency of each word
word_counts = Counter(all_words)

# Convert the Counter to a pandas DataFrame for easier analysis
word_freq_df = pd.DataFrame(
    word_counts.items(), columns=["word", "frequency"]
).sort_values(by="frequency", ascending=False)

print("--- Word Frequency Analysis ---")
print(word_freq_df.head(10))
print("\n")
print(word_freq_df.tail(10))

# --- 4. Visualize the Most Frequent Words ---
plt.figure(figsize=(10, 6))
top_20_words = word_freq_df.head(20)
plt.bar(top_20_words["word"], top_20_words["frequency"])
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 20 Most Frequent Words")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# --- 5. Filter and Finalize ---
# Decide on the frequency thresholds
# For example, remove words that appear more than 1000 times or only once.
# For this small dataset, let's remove words that appear > 3 times or only 1 time.
MIN_FREQ = 2
MAX_FREQ = 3

# Get the list of words to remove
words_to_remove = set(
    word_freq_df[
        (word_freq_df["frequency"] < MIN_FREQ) | (word_freq_df["frequency"] > MAX_FREQ)
    ]["word"]
)

print(f"\n--- Filtering ---")
print(f"Removing {len(words_to_remove)} words based on frequency.")
print(f"Words to remove: {words_to_remove}")

# Create a final, filtered list of tokens for each narrative
df["filtered_tokens"] = df["tokens"].apply(
    lambda tokens: [word for word in tokens if word not in words_to_remove]
)

# Join the tokens back into a string, ready for vectorization
df["processed_narrative"] = df["filtered_tokens"].apply(lambda tokens: " ".join(tokens))

print("\n--- Final Processed Data ---")
print(df[["narrative", "processed_narrative"]].head())
