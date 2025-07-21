import json
import logging
import os

import data_tools
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

SCORING = 'roc_auc'
N_SPLITS = 5
CLF_SRC_DIR = os.path.split(os.path.abspath(__file__))[0]

def logistic_regression(
    xtr_df, ytr_df, ktr_df, 
    xte_df, yte_df, kte_df, 
    output_dir, algo_list, seed
):
    preproc_name = 'CountVectorizer'
    classifier_name = 'logistic_regression'
    pipe, clf_tuning_kws = make_clf_pipeline(
        preproc_name, classifier_name, memory_dir=output_dir
    )
    kfold_cv = sklearn.model_selection.StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=seed
    )

    searcher = sklearn.model_selection.GridSearchCV(
        pipe,
        clf_tuning_kws,
        scoring=SCORING,
        cv=kfold_cv,
        return_train_score=True,
        refit=True,
        error_score="raise",
    )
    # the grid search expects a column vector of labels
    x_vals = [str(doc) for doc in xtr_df.values[:,0]]
    y_vals = ytr_df.values[:, 0]
    searcher.fit(x_vals, y_vals)
    logging.info("Best parameters found: %s", searcher.best_params_)
    cv_tr_perf_df, cv_va_perf_df = extract_perf_df_from_cv_obj(searcher)
    out_csv_path = os.path.join(
        output_dir,
        "perf_by_hyper_%dfoldCV_seed%d_{split}.csv" % (N_SPLITS, seed),
    )
    cv_tr_perf_df.to_csv(out_csv_path.format(split="traincv"), index=False)
    cv_va_perf_df.to_csv(out_csv_path.format(split="validcv"), index=False)
    logging.info("Wrote hyper search info to: %s", out_csv_path)

    xte_vals = [str(doc) for doc in xte_df.values[:, 0]]
    yte_vals = yte_df.values[:, 0]
    yte_pred = searcher.best_estimator_.predict(xte_vals)
    yte_pred_proba = searcher.best_estimator_.predict_proba(xte_vals)[:, 1]

    conf_matrix = sklearn.metrics.confusion_matrix(yte_vals, yte_pred)
    logging.info("Confusion Matrix:\n%s", conf_matrix)  
    # @TODO: save confusion matrix to file
    # @TODO: save roc-auc to file
    
    # score with scoring value
    test_score = sklearn.metrics.get_scorer(SCORING)(
        searcher.best_estimator_, xte_vals, yte_vals
    )

    best_estimator_results = {
        "best_params": searcher.best_params_,
        "test_score": test_score,
    }

    best_estimator_df = pd.DataFrame([best_estimator_results])
    best_estimator_file_path = os.path.join(output_dir, "best_estimator_results.csv")
    best_estimator_df.to_csv(best_estimator_file_path, index=False)
    logging.info("Wrote best estimator results to: %s", best_estimator_file_path)


def naive_bayes(
    xtr_df, ytr_df, ktr_df, xte_df, yte_df, kte_df, output_dir, algo_list, seed
):

    pass  # Implement the Naive Bayes logic here


def random_forest(
    xtr_df, ytr_df, ktr_df, xte_df, yte_df, kte_df, output_dir, algo_list, seed
):
    pass  # Implement the Random Forest logic here


def xgboost(
    xtr_df, ytr_df, ktr_df, xte_df, yte_df, kte_df, output_dir, algo_list, seed
):
    pass  # Implement the XGBoost logic here


# TODO: implement ngram support
# TODO: implement vectorizer support
def make_clf_pipeline(
    preproc_name, clf_name, memory_dir=None, vectorizer=None, ngram_range=None):
    """
    Creates a scikit-learn pipeline for a classifier with specified preprocessing and classifier configurations.

    Parameters:
    preproc_name (str): The name of the preprocessing technique to use. Supported values are "countvec" for CountVectorizer and "tfidf" for TfidfVectorizer.
    clf_name (str): The name of the classifier configuration file (without the .json extension) located in the CLF_SRC_DIR directory.
    memory_dir (str, optional): The directory to use for caching the pipeline. Defaults to None.
    vectorizer (sklearn vectorizer, optional): Pre-trained vectorizer to use instead of creating a new one. If provided, preproc_name is ignored.
    ngram_range (tuple, optional): The ngram range to use (e.g., (1,1) for unigrams, (1,2) for uni+bigrams). Only used if vectorizer is None.

    Returns:
    tuple: A tuple containing:
        - sklearn.pipeline.Pipeline: The constructed scikit-learn pipeline with the specified preprocessing and classifier.
        - dict: A dictionary of classifier tuning keywords for hyperparameter tuning.
    """
    logging.info("Creating classifier pipeline with preproc: %s, clf: %s", preproc_name, clf_name)
    steps = list()

    # Use pre-trained vectorizer if provided, otherwise create new one
    if vectorizer is not None:
        logging.info("Using pre-trained vectorizer")
        steps.append(("txt2vec", vectorizer))
    else:
        # Create new vectorizer with optional ngram_range
        vectorizer_kwargs = {}
        if ngram_range is not None:
            vectorizer_kwargs['ngram_range'] = ngram_range
            logging.info("Using ngram_range: %s", ngram_range)   
                 
        if preproc_name.lower().count("countvec"):
            steps.append(("txt2vec", sklearn.feature_extraction.text.CountVectorizer(**vectorizer_kwargs)))
        elif preproc_name.lower().count("tfidf"):
            steps.append(("txt2vec", sklearn.feature_extraction.text.TfidfVectorizer(**vectorizer_kwargs)))

    json_path = os.path.join(CLF_SRC_DIR, clf_name + ".json")
    with open(json_path, "r") as f:
        clf_dict = json.load(f)
    clf_kws = {}
    clf_tuning_kws = {}
    for k, v in clf_dict.items():
        if k.startswith("CLF__"):
            if k.startswith("CLF__grid"):
                clf_tuning_kws["clf" + k[9:]] = v
            continue
        clf_kws[k] = v

    constructor_str = clf_dict["CLF__constructor"]
    assert constructor_str.startswith("sklearn")
    for ii, name in enumerate(constructor_str.split(".")):
        if ii == 0:
            mod = globals().get(name)
        else:
            mod = getattr(mod, name)
    steps.append(("clf", mod(**clf_kws)))
    pipe = sklearn.pipeline.Pipeline(steps, memory=memory_dir)
    return pipe, clf_tuning_kws


def extract_perf_df_from_cv_obj(searcher):
    """
    Extracts performance DataFrames from a cross-validation search object.
    This function takes a cross-validation search object and extracts two DataFrames:
    one for training performance and one for testing performance. It also separates
    the hyperparameter values into individual columns.
    Args:
        searcher (object): A cross-validation search object with `cv_results_` attribute.
    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - cv_tr_perf_df (DataFrame): DataFrame containing training performance metrics.
            - cv_te_perf_df (DataFrame): DataFrame containing testing performance metrics.
    """
    cv_perf_df = pd.DataFrame(searcher.cv_results_)

    # Make separate columns for each hyper value
    hypers_df = pd.DataFrame(cv_perf_df["params"].values.tolist())
    hkeys = [c for c in hypers_df.columns]
    hypers_df["params"] = cv_perf_df["params"].values.copy()
    hkeys += ["params"]

    tr_split_keys = ["mean_train_score"] + [
        "split%d_train_score" % a for a in range(searcher.n_splits_)
    ]
    te_split_keys = ["mean_test_score"] + [
        "split%d_test_score" % a for a in range(searcher.n_splits_)
    ]
    cv_tr_perf_df = cv_perf_df[tr_split_keys].copy()
    cv_te_perf_df = cv_perf_df[te_split_keys].copy()
    cv_tr_perf_df.rename(
        dict(zip(tr_split_keys, [a.replace("_train", "") for a in tr_split_keys])),
        axis="columns",
        inplace=True,
    )
    cv_te_perf_df.rename(
        dict(zip(te_split_keys, [a.replace("_test", "") for a in te_split_keys])),
        axis="columns",
        inplace=True,
    )
    okeys = [c for c in cv_tr_perf_df.columns]
    cv_tr_perf_df[hypers_df.columns] = hypers_df.values
    cv_te_perf_df[hypers_df.columns] = hypers_df.values

    # Reorder cols
    cv_tr_perf_df = cv_tr_perf_df[hkeys + okeys].copy()
    cv_te_perf_df = cv_te_perf_df[hkeys + okeys].copy()

    return cv_tr_perf_df, cv_te_perf_df
