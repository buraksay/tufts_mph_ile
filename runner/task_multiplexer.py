import logging

import clf_tools
import config
import data_tools
import experiments


def preprocess_data(task_id, task_tag, output_dir, algo_list, df):
    logging.info(
        "Preprocessing data for task %s with tag %s and algorithms %s",
        task_id,
        task_tag,
        algo_list,
    )
    # breakpoint()
    logging.info("applying clean")
    df["Narrative"] = df["Narrative"].apply(data_tools.clean)
    data_tools.save_csv_file(df, "x_df_clean.csv", output_dir=output_dir)

    # TODO: fork for custom_stopwords
    stopword_func = get_stopword_func(algo_list)
    logging.info("applying stopwords with function: %s", stopword_func.__name__)
    df["Narrative"] = df["Narrative"].apply(stopword_func)
    data_tools.save_csv_file(df, "x_df_stopwords.csv", output_dir=output_dir)

    # TODO: fork for snowball_stemmer
    stemmer_func = get_stemmer_func(algo_list)
    logging.info("applying stemmer with function: %s", stemmer_func.__name__)
    df["Narrative"] = df["Narrative"].apply(stemmer_func)
    data_tools.save_csv_file(df, "x_df_stemmer.csv", output_dir=output_dir)

    logging.debug("Preprocessed DataFrame head: %s", df.head())
    return df


def split_data(x_df, y_df, k_df, split_date, seed, output_dir):
    logging.info(
        "Splitting data into train and test sets with split date %s and seed %d",
        split_date,
        seed,
    )
    (xtr_df, ytr_df, ktr_df), (xtest_df, ytest_df, ktest_df) = (
        data_tools.split_into_train_test(
            xdf=x_df, ydf=y_df, kdf=k_df, split_date=split_date, random_state=seed
        )
    )
    for [df, filename] in [
        [xtr_df, "x_train.csv"],
        [ytr_df, "y_train.csv"],
        [ktr_df, "key_train.csv"],
        [xtest_df, "x_test.csv"],
        [ytest_df, "y_test.csv"],
        [ktest_df, "key_test.csv"],
    ]:
        data_tools.save_csv_file(df, filename, output_dir=output_dir)

    return (xtr_df, ytr_df, ktr_df), (xtest_df, ytest_df, ktest_df)


def train_classifier(
    xtr_df, ytr_df, ktr_df, xte_df, yte_df, kte_df, algo_list, output_dir, seed
):
    logging.info("Training classifier... %s", algo_list)
    get_classifier_func(algo_list)(
        xtr_df=xtr_df,
        ytr_df=ytr_df,
        ktr_df=ktr_df,
        xte_df=xte_df,
        yte_df=yte_df,
        kte_df=kte_df,
        output_dir=output_dir,
        algo_list=algo_list,
        seed=seed,
        vectorizer=get_vectorizer_func(algo_list),
        ngram_range=get_ngram(algo_list),
    )


def get_stopword_func(algo_list):
    if "sw.csw" in algo_list:
        logging.info("Using custom stopwords function")
        return data_tools.custom_stopwords
    elif "sw.std" in algo_list:
        logging.info("Using standard stopwords function")
        return data_tools.standard_stopwords
    elif "sw.pun" in algo_list:
        logging.info("Using punctuation removal function")
        return data_tools.remove_punctuation
    else:
        raise ValueError(
            "No valid stopword function found in the algorithm list: %s" % algo_list
        )


def get_vectorizer_func(algo_list):
    if "vec.tfidf" in algo_list:
        logging.info("Using TF-IDF vectorizer")
        return config.TFIDF
    elif "vec.bow" in algo_list:
        logging.info("Using Bag of Words vectorizer")
        return config.BAG_OF_WORDS
    else:
        raise ValueError(
            "No valid vectorizer function found in the algorithm list: %s" % algo_list
        )


def get_stemmer_func(algo_list):
    if "stem.snow" in algo_list:
        logging.info("Using Snowball stemmer")
        return data_tools.snowball_stemmer
    elif "stem.por" in algo_list:
        logging.info("Using Porter stemmer")
        return data_tools.porter_stemmer
    else:
        raise ValueError(
            "No valid stemmer function found in the algorithm list: %s" % algo_list
        )


def get_ngram(algo_list):
    if "ngram.uni" in algo_list:
        logging.info("Using Unigram model")
        return config.UNIGRAM
    elif "ngram.bi" in algo_list:
        logging.info("Using Bigram model")
        return config.BIGRAM
    elif "ngram.tri" in algo_list:
        logging.info("Using Trigram model")
        return config.TRIGRAM
    else:
        raise ValueError("No valid n-gram found in the algorithm list: %s" % algo_list)


def get_classifier_func(algo_list):
    if "clf.nb" in algo_list:
        logging.info=("Using Naive Bayes classifier")
        return clf_tools.naive_bayes_classifier
    elif "clf.lr" in algo_list:
        logging.info("Using Logistic Regression classifier")
        return clf_tools.logistic_regression
    elif "clf.xgb" in algo_list:
        logging.info("Using XGBoost classifier")
        return clf_tools.xgboost_classifier
    elif "clf.rf" in algo_list:
        logging.info("Using Random Forest classifier")
        return clf_tools.random_forest_classifier
    else:
        raise ValueError(
            "No valid classifier function found in the algorithm list: %s" % algo_list
        )
