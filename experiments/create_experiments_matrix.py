import itertools

import pandas as pd

# Define mappings from long-form to abbreviation
algo_mappings = {
    "Stopwords": {
        "sw.std": {"name": "Standard Stopwords", "func": "standard_stopwords"},
        "sw.csw": {"name": "Custom Stopwords", "func": "custom_stopwords"},
        "sw.pun": {"name": "Remove Punctuation", "func": "remove_punctuation"},
    },
    "Stemmer": {
        "stem.por": {"name": "Porter Stemmer", "func": "porter_stemmer"},
        "stem.snow": {"name": "Snowball Stemmer", "func": "snowball_stemmer"},
    },
    "Ngram": {
        "ngram.uni": {"name": "Uni-gram", "func": "uni_gram"},
        "ngram.bi": {"name": "Bi-gram", "func": "bi_gram"},
        "ngram.tri": {"name": "Tri-gram", "func": "tri_gram"},
    },
    "Vectorizer": {
        "vec.tfidf": {"name": "TF-IDF", "func": "tfidf"},
        "vec.bow": {"name": "Bag of Words", "func": "bag_of_words"},
    },
    "Classifier": {
        "clf.lr": {"name": "Logistic Regression", "func": "logistic_regression"},
        "clf.nb": {"name": "Naive Bayes", "func": "naive_bayes"},
        "clf.rf": {"name": "Random Forest", "func": "random_forest"},
        "clf.xgb": {"name": "XGBoost", "func": "xgboost"},
    },
}

# Create Tag using the mappings
def create_tag(row):
    tag_parts = []
    for dimension_name in dimensions.keys():
        long_form_name = row[dimension_name]
        abbrev = mappings[dimension_name][long_form_name]
        tag_parts.append(abbrev)
    return "-".join(tag_parts)


if __name__ == '__main__':
    # Extract long-form options and mappings from algo_mappings
    dimensions = {}
    mappings = {}

    for dimension_name, options in algo_mappings.items():
        # Extract the long-form names
        dimensions[dimension_name] = [
            option_data["name"] for option_data in options.values()
        ]
        # Create mapping from long-form name to abbreviation
        mappings[dimension_name] = {
            option_data["name"]: abbrev for abbrev, option_data in options.items()
        }

    # Create combinations
    combinations = list(itertools.product(*dimensions.values()))

    # Create DataFrame with ID and Tag
    df = pd.DataFrame(combinations, columns=list(dimensions.keys()))
    df["ID"] = ["EXP{:03d}".format(i + 1) for i in range(len(df))]

    df["Tag"] = df.apply(create_tag, axis=1)

    # Rearrange columns for readability
    column_order = ["ID", "Tag"] + list(dimensions.keys())
    df = df[column_order]

    # Save for later use
    df.to_csv("nlp_experiment_matrix2.csv", index=False)
    print(df.head())
    print(f"\nTotal experiments: {len(df)}")
