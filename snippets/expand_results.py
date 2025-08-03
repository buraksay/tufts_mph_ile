import os

import pandas as pd

algo_mappings = {
    "Stopwords": {
        "sw.std": {"name": "StdSW"},
        "sw.csw": {"name": "CustSW"},
        "sw.pun": {"name": "PuncSW"},
    },
    "Stemmer": {
        "stem.por": {"name": "Porter"},
        "stem.snow": {"name": "Snowball"},
    },
    "Ngram": {
        "ngram.uni": {"name": "1-gram"},
        "ngram.bi": {"name": "2-gram"},
        "ngram.tri": {"name": "3-gram"},
    },
    "Vectorizer": {
        "vec.tfidf": {"name": "TF-IDF"},
        "vec.bow": {"name": "BoW"},
    },
    "Classifier": {
        "clf.lr": {"name": "LogReg"},
        "clf.xgb": {"name": "XGBoost"},
        "clf.nb": {"name": "NaiveBayes"},
        "clf.rf": {"name": "RandFor"},
    },
}

def expand_results():
    """
    Expands the task_tag column in the results CSV file into separate columns
    with human-friendly names for each technique component.
    """
    results_dir = "/Users/baba/proj/aphaproject/tufts_mph_ile/results/newres/"
    input_file = os.path.join(results_dir, "combined_results_20250803.csv")
    output_file = os.path.join(results_dir, "expanded_results_20250803.csv")
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {input_file}")
    except FileNotFoundError:
        print(f"Error: File not found at {input_file}")
        return
    abbrev_to_name = {}
    for category, options in algo_mappings.items():
        for abbrev, option_data in options.items():
            abbrev_to_name[abbrev] = option_data["name"]
    df["Stopwords"] = ""
    df["sw"] = ""
    df["Stemming"] = ""
    df["stem"] = ""
    df["Ngram"] = ""
    df["ngram"] = ""
    df["Vectorizer"] = ""
    df["vec"] = ""
    df["Classifier"] = ""
    df["clf"] = ""
    for idx, row in df.iterrows():
        tag = row["task_tag"]
        tag_parts = tag.split("-")

        for part in tag_parts:
            if part.startswith("sw."):
                abbrev = part
                df.at[idx, "Stopwords"] = abbrev_to_name.get(abbrev, "Unknown")
                df.at[idx, "sw"] = part.split(".")[1]
            elif part.startswith("stem."):
                abbrev = part
                df.at[idx, "Stemming"] = abbrev_to_name.get(abbrev, "Unknown")
                df.at[idx, "stem"] = part.split(".")[1]
            elif part.startswith("ngram."):
                abbrev = part
                df.at[idx, "Ngram"] = abbrev_to_name.get(abbrev, "Unknown")
                df.at[idx, "ngram"] = part.split(".")[1]
            elif part.startswith("vec."):
                abbrev = part
                df.at[idx, "Vectorizer"] = abbrev_to_name.get(abbrev, "Unknown")
                df.at[idx, "vec"] = part.split(".")[1]
            elif part.startswith("clf."):
                abbrev = part
                df.at[idx, "Classifier"] = abbrev_to_name.get(abbrev, "Unknown")
                df.at[idx, "clf"] = part.split(".")[1]
    desired_column_order = [
        "task_id", 
        "Stopwords", "Stemming", "Ngram", "Vectorizer", "Classifier",
        "f1_score", "roc_auc", "precision", "recall", "accuracy",
        "sw", "stem", "ngram", "vec", "clf", 
        "task_tag"
    ]
    for col in desired_column_order:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame")
    available_columns = [col for col in desired_column_order if col in df.columns]
    df = df[available_columns]
    df.to_csv(output_file, index=False, mode="w+")
    print(f"Successfully expanded results and saved to {output_file}")
    print(f"Total rows processed: {len(df)}")
    print("Expanded columns:")
    print(df.columns.tolist())

if __name__ == "__main__":
    expand_results()
