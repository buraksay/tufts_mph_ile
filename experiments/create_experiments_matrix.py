import argparse
import itertools

import config
import data_tools
import pandas as pd

# Define mappings from long-form to abbreviation
algo_mappings = {
    "Stopwords": {
        config.SW_STD: {"name": "Standard Stopwords", "func": config.STANDARD_STOPWORDS},
        config.SW_CSW: {"name": "Custom Stopwords", "func": config.CUSTOM_STOPWORDS},
        config.SW_PUN: {"name": "Remove Punctuation", "func": config.REMOVE_PUNCTUATION},
    },
    "Stemmer": {
        config.STEM_POR: {"name": "Porter Stemmer", "func": config.PORTER_STEMMER},
        config.STEM_SNOW: {"name": "Snowball Stemmer", "func": config.SNOWBALL_STEMMER},
    },
    "Ngram": {
        config.NGRAM_UNI: {"name": "Uni-gram", "func": config.UNIGRAM},
        config.NGRAM_BI: {"name": "Bi-gram", "func": config.BIGRAM},
        config.NGRAM_TRI: {"name": "Tri-gram", "func": config.TRIGRAM},
    },
    "Vectorizer": {
        config.VEC_TFIDF: {"name": "TF-IDF", "func": config.TFIDF},
        config.VEC_BOW: {"name": "Bag of Words", "func": config.BAG_OF_WORDS},
    },
    "Classifier": {
        config.CLF_LR: {"name": "Logistic Regression", "func": config.LOGISTIC_REGRESSION},
        config.CLF_NB: {"name": "Naive Bayes", "func": config.NAIVE_BAYES},
        config.CLF_RF: {"name": "Random Forest", "func": config.RANDOM_FOREST},
        config.CLF_XGB: {"name": "XGBoost", "func": config.XGBOOST},
    },
}
# Create Tag using the mappings
def create_tag(row, dimensions, mappings):
    tag_parts = []
    for dimension_name in dimensions.keys():
        long_form_name = row[dimension_name]
        abbrev = mappings[dimension_name][long_form_name]
        tag_parts.append(abbrev)
    return "-".join(tag_parts)


def create_experiment_matrix(out_file, output_dir):
    """
    Create a matrix of NLP experiments based on the defined configurations.
    """
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

    df["Tag"] = df.apply(create_tag, args=(dimensions,mappings), axis=1)

    # Rearrange columns for readability
    column_order = ["ID", "Tag"] + list(dimensions.keys())
    df = df[column_order]

    # Save for later use
    data_tools.save_csv_file(df, out_file, output_dir=output_dir, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a matrix of NLP experiments."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="nlp_experiment_matrix2.csv",
        help="Output CSV file name (default: nlp_experiment_matrix2.csv)",
    )
    args = parser.parse_args()


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

    df["Tag"] = df.apply(create_tag, args=(dimensions,mappings), axis=1)

    # Rearrange columns for readability
    column_order = ["ID", "Tag"] + list(dimensions.keys())
    df = df[column_order]

    # Save for later use
    df.to_csv(args.out, index=False)
    print(df.head())
    print(f"\nTotal experiments: {len(df)}")
