import logging
import sys

import data_tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

STANDARD_SW = set(stopwords.words('english'))
# Top 40 words by document frequency from the corpus, post
CUSTOM_SW = set(
    [
        "dispatched",
        "report",
        "pt",
        "care",
        "stretcher",
        "secured",
        "arrival",
        "bed",
        "noted",
        "given",
        "transport",
        "rn",
        "transferred",
        "scene",
        "clear",
        "vitals",
        "narcan",
        "incident",
        "end",
        "yo",
        "skin",
        "airway",
        "hospital",
        "rails",
        "heroin",
        "male",
        "mg",
        "er",
        "pupils",
        "unresponsive",
        "patent",
        "assisted",
        "route",
        "breathing",
        "monitored",
        "crew",
        "pain",
        "en",
        "straps",
    ]
) | STANDARD_SW

def apply_standard_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in STANDARD_SW]
    return ' '.join(tokens)

def standard_stopwords(df, column='Narrative'):
    """
    Apply standard stopwords removal to a DataFrame column.
    """
    df[column] = df[column].apply(apply_standard_stopwords)
    return df[column]


def apply_custom_stopwords(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in CUSTOM_SW]
    return ' '.join(tokens)

def custom_stopwords(df, column='Narrative'):
    """
    Apply custom stopwords removal to a DataFrame column.
    """
    df[column] = df[column].apply(apply_custom_stopwords)
    return df[column]


# noop
def remove_punctuation(df, column="Narrative"):
    return df[column]


TOP_COUNT = 50
BOTTOM_COUNT = 50


def run_frequency_analysis(df, output_dir):
    # Apply basic cleaning
    df["cleaned_narrative"] = df["Narrative"].apply(basic_text_cleaning)
    # Analyze frequencies
    freq_df, vectorizer = analyze_word_frequencies(df["cleaned_narrative"].tolist())

    # Save the frequency DataFrame
    data_tools.save_csv_file(freq_df, "word_frequencies.csv", output_dir=output_dir)

    logging.info(
        f"Top %d most frequent words:\n%s",
        TOP_COUNT,
        freq_df.sort_values("doc_percentage", ascending=False).head(TOP_COUNT),
    )
    logging.info(
        f"\nBottom %d least frequent words:\n%s",
        BOTTOM_COUNT,
        freq_df.tail(BOTTOM_COUNT),
    )
    plot_frequency_analysis(freq_df)

    rare_words, common_words = analyze_thresholds(freq_df, len(df))
    # Apply filtering (adjust thresholds based on your analysis)
    filtered_vectorizer, kept_words = filter_vocabulary(
        freq_df,
        df["cleaned_narrative"].tolist(),
        min_freq_threshold=2,  # Remove words appearing in fewer than 2 documents
        max_doc_freq_threshold=0.8,  # Remove words appearing in more than 80% of documents
    )
    filtered_vocabulary = set(filtered_vectorizer.get_feature_names_out())

    # Apply to your data
    df["final_processed"] = df["Narrative"].apply(
        lambda x: final_text_preprocessing(x, filtered_vocabulary)
    )
    logging.info("\nSample of final processed texts:")
    for i in range(min(3, len(df))):
        logging.info(f"Original: {df.iloc[i]['Narrative'][:100]}...")
        logging.info(f"Processed: {df.iloc[i]['final_processed'][:100]}...")
        logging.info("-" * 50)

    return filtered_vectorizer, kept_words


# --- 2. Calculate Word Frequencies ---
def analyze_word_frequencies(texts, min_df=1, max_df=1.0, ngram_range=(1, 1)):
    """
    Analyze word frequencies in your corpus

    Parameters:
    - texts: list of text documents
    - min_df: ignore terms that appear in fewer than min_df documents
    - max_df: ignore terms that appear in more than max_df fraction of documents
    - ngram_range: tuple for n-gram range (1,1) for unigrams, (1,2) for uni+bigrams
    """

    # Use CountVectorizer to get word frequencies
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words="english",  # removes common English stopwords
    )

    # Fit the vectorizer
    word_matrix = vectorizer.fit_transform(texts)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Calculate total frequency for each word across all documents
    word_frequencies = np.array(word_matrix.sum(axis=0)).flatten()

    # Calculate document frequency (how many documents contain each word)
    doc_frequencies = np.array((word_matrix > 0).sum(axis=0)).flatten()

    # Create frequency DataFrame
    freq_df = pd.DataFrame(
        {
            "word": feature_names,
            "total_frequency": word_frequencies,
            "document_frequency": doc_frequencies,
            "doc_percentage": (doc_frequencies / len(texts)) * 100,
        }
    ).sort_values("total_frequency", ascending=False)

    return freq_df, vectorizer


# --- 3. Visualize Frequency Distributions ---
def plot_frequency_analysis(freq_df):
    """Create visualizations for word frequency analysis"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Top 20 most frequent words
    top_20 = freq_df.head(TOP_COUNT)
    axes[0, 0].bar(range(len(top_20)), top_20["total_frequency"])
    axes[0, 0].set_xticks(range(len(top_20)))
    axes[0, 0].set_xticklabels(top_20["word"], rotation=45, ha="right")
    axes[0, 0].set_title("Top %d Most Frequent Words" % TOP_COUNT)
    axes[0, 0].set_ylabel("Total Frequency")

    # Plot 2: Frequency distribution (log scale)
    axes[0, 1].hist(freq_df["total_frequency"], bins=50, edgecolor="black")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Word Frequency Distribution (Log Scale)")
    axes[0, 1].set_xlabel("Total Frequency")
    axes[0, 1].set_ylabel("Number of Words (Log Scale)")

    # Plot 3: Document frequency vs total frequency
    axes[1, 0].scatter(
        freq_df["document_frequency"], freq_df["total_frequency"], alpha=0.6
    )
    axes[1, 0].set_xlabel("Document Frequency")
    axes[1, 0].set_ylabel("Total Frequency")
    axes[1, 0].set_title("Document Frequency vs Total Frequency")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")

    # Plot 4: Cumulative frequency
    freq_df_sorted = freq_df.sort_values("total_frequency", ascending=False)
    cumulative_freq = freq_df_sorted["total_frequency"].cumsum()
    axes[1, 1].plot(range(len(cumulative_freq)), cumulative_freq)
    axes[1, 1].set_title("Cumulative Word Frequency")
    axes[1, 1].set_xlabel("Word Rank")
    axes[1, 1].set_ylabel("Cumulative Frequency")

    plt.tight_layout()
    plt.show()


# --- 4. Determine Filtering Thresholds ---
def analyze_thresholds(freq_df, total_documents):
    """Help determine good filtering thresholds"""

    logging.info(f"Total unique words: {len(freq_df)}")
    logging.info(f"Total documents: {total_documents}")

    # Words that appear in very few documents (potential noise)
    rare_words = freq_df[freq_df["document_frequency"] == 1]
    logging.info(
        f"Words appearing in only 1 document: {len(rare_words)} ({len(rare_words)/len(freq_df)*100:.1f}%)"
    )

    # Words that appear in most documents (potential generic terms)
    common_words = freq_df[freq_df["doc_percentage"] > 80]
    logging.info(
        f"Words appearing in >80% of documents: {len(common_words)} ({len(common_words)/len(freq_df)*100:.1f}%)"
    )

    # Frequency percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    logging.info(f"\nFrequency percentiles:")
    for p in percentiles:
        threshold = np.percentile(freq_df["total_frequency"], p)
        logging.info(f"{p}th percentile: {threshold:.0f}")

    return rare_words, common_words


# --- 5. Apply Filtering ---
def filter_vocabulary(freq_df, texts, min_freq_threshold=2, max_doc_freq_threshold=0.8):
    """
    Create filtered vocabulary based on frequency thresholds

    Parameters:
    - min_freq_threshold: minimum total frequency to keep a word
    - max_doc_freq_threshold: maximum document frequency percentage (0-1) to keep a word
    """

    # Analyze with filtering
    vectorizer = CountVectorizer(
        min_df=min_freq_threshold,  # minimum document frequency
        max_df=max_doc_freq_threshold,  # maximum document frequency
        stop_words="english",
    )

    # Fit the vectorizer
    word_matrix = vectorizer.fit_transform(texts)
    kept_words = vectorizer.get_feature_names_out()

    logging.info(f"Original vocabulary size: {len(freq_df)}")
    logging.info(f"Filtered vocabulary size: {len(kept_words)}")
    logging.info(
        f"Removed: {len(freq_df) - len(kept_words)} words ({(len(freq_df) - len(kept_words))/len(freq_df)*100:.1f}%)"
    )

    return vectorizer, kept_words


def basic_text_cleaning(text):
    """Basic text cleaning before frequency analysis"""
    # Convert to lowercase
    text = str(text).lower()
    # Remove extra whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()
    # Remove numbers and punctuation (optional - depends on your needs)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# --- 6. Final Preprocessing Function ---
def final_text_preprocessing(text, vocabulary):
    """Final preprocessing using the filtered vocabulary"""
    cleaned_text = basic_text_cleaning(text)
    tokens = word_tokenize(cleaned_text)
    filtered_tokens = [word for word in tokens if word in vocabulary]

    # Transform using the fitted vectorizer
    # This will automatically filter out words not in the vocabulary
    return " ".join(filtered_tokens)
