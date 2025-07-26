import json
import logging
import os

import config
import data_tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, roc_curve)
from sklearn.naive_bayes import MultinomialNB

CLF_SRC_DIR = os.path.split(os.path.abspath(__file__))[0]

def naive_bayes(
    xtr_df, ytr_df, ktr_df, 
    xte_df, yte_df, kte_df, 
    output_dir, algo_list, seed,
    vectorizer, ngram_range):
    '''
    vectorizer = CountVectorizer()
    X_train_bow = vectorizer.fit_transform(xtr_df)
    X_test_bow = vectorizer.transform(xte_df)

    # Train a Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_bow, ytr_df)

    # Predict probabilities for the test set
    y_probs = model.predict_proba(X_test_bow)[:, 1]  # Get the probability for the positive class
    y_pred = model.predict(X_test_bow)

    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(yte_df, y_probs)

    roc_auc = auc(fpr, tpr)
    logging.info(accuracy = accuracy_score(yte_df, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(yte_df, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(yte_df, y_pred)
    
    # Assign values for readability
    tn, fp, fn, tp = cm.ravel()
    
    # Print confusion matrix with custom labels
    print("Confusion Matrix:")
    print(f"                Predicted ORI    Predicted False ORI")
    print(f"True ORI        {tp:<18} {fn}")
    print(f"False ORI       {fp:<18} {tn}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUROC: BOW NaiveBayes')
    plt.legend(loc='lower right')
    plt.show()
    '''
    logging.info("Training Naive Bayes classifier...")
    classifier_name = 'naive_bayes'
    return classify_with_kfold_cv(xtr_df, ytr_df, ktr_df,
        xte_df, yte_df, kte_df,
        output_dir, algo_list, seed,
        vectorizer, ngram_range, classifier_name)



def classify_with_kfold_cv(xtr_df, ytr_df, ktr_df,
    xte_df, yte_df, kte_df,
    output_dir, algo_list, seed,
    vectorizer, ngram_range, classifier_name):

    perf_metrics = set()
    
    # if using pre-trained vectorizer, pass it here
    pipe, clf_tuning_kws = make_clf_pipeline(
        vectorizer, classifier_name, memory_dir=None, vectorizer=None, ngram_range=ngram_range
        # vectorizer, classifier_name, memory_dir=output_dir, vectorizer=None, ngram_range=ngram_range
    )
    kfold_cv = sklearn.model_selection.StratifiedKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=seed
    )
    searcher = sklearn.model_selection.GridSearchCV(
        pipe,
        clf_tuning_kws,
        scoring=config.SCORING,
        cv=kfold_cv,
        return_train_score=True,
        refit=True,
        error_score="raise",
        verbose=3
    )
    # the grid search expects a column vector of labels
    x_vals = [str(doc) for doc in xtr_df.values[:,0]]
    y_vals = ytr_df.values[:, 0]
    searcher.fit(x_vals, y_vals)
    logging.info("Best parameters found: %s", searcher.best_params_)
    cv_tr_perf_df, cv_va_perf_df = extract_perf_df_from_cv_obj(searcher)
    out_csv_file = "perf_by_hyper_%dfoldCV_seed%d_{split}.csv" % (config.N_SPLITS, seed)
    out_csv_path = os.path.join(
        output_dir, out_csv_file
    )
    cv_tr_perf_df.to_csv(out_csv_path.format(split="train_cross_val"), index=False)
    cv_va_perf_df.to_csv(out_csv_path.format(split="test_cross_val"), index=False)
    logging.info("Wrote hyper search info to: %s", out_csv_path)

    save_fit_tokens(searcher.best_estimator_, output_dir)

    xte_vals = [str(doc) for doc in xte_df.values[:, 0]]
    yte_vals = yte_df.values[:, 0]
    yte_pred = searcher.best_estimator_.predict(xte_vals)
    yte_pred_proba = searcher.best_estimator_.predict_proba(xte_vals)[:, 1]

    conf_matrix = sklearn.metrics.confusion_matrix(yte_vals, yte_pred)
    logging.info("Confusion Matrix:\n%s", conf_matrix)  
    # @TODO: save confusion matrix to file
    # @TODO: save roc-auc to file

    # score with scoring value
    test_score = sklearn.metrics.get_scorer(config.SCORING)(
        searcher.best_estimator_, xte_vals, yte_vals
    )
    f1_score = sklearn.metrics.f1_score(yte_vals, yte_pred)
    roc_auc = sklearn.metrics.roc_auc_score(yte_vals, yte_pred_proba)
    precision = sklearn.metrics.precision_score(yte_vals, yte_pred)
    recall = sklearn.metrics.recall_score(yte_vals, yte_pred)
    accuracy = sklearn.metrics.accuracy_score(yte_vals, yte_pred)
    perf_metrics = {
        config.F1_SCORE: f1_score,
        config.ROC_AUC: roc_auc,
        config.PRECISION: precision,
        config.RECALL: recall,
        config.ACCURACY: accuracy
    }

    best_estimator_results = {
        "best_params": searcher.best_params_,
        "test_score": test_score,
    }

    best_estimator_df = pd.DataFrame([best_estimator_results])
    best_estimator_file_path = os.path.join(output_dir, "best_estimator_results.csv")
    best_estimator_df.to_csv(best_estimator_file_path, index=False)
    logging.info("Wrote best estimator results to: %s", best_estimator_file_path)
    logging.info("Test score: %f", test_score)

    return perf_metrics


def random_forest(xtr_df, ytr_df, ktr_df,
    xte_df, yte_df, kte_df,
    output_dir, algo_list, seed,
    vectorizer, ngram_range):

    logging.info("Training Random Forest classifier...")
    classifier_name = 'random_forest'
    return classify_with_kfold_cv(xtr_df, ytr_df, ktr_df,
        xte_df, yte_df, kte_df,
        output_dir, algo_list, seed,
        vectorizer, ngram_range, classifier_name)


def xgboost(xtr_df, ytr_df, ktr_df,
    xte_df, yte_df, kte_df,
    output_dir, algo_list, seed,
    vectorizer, ngram_range):
    
    logging.info("Training XGBoost classifier...")
    classifier_name = 'xgboost'
    return classify_with_kfold_cv(xtr_df, ytr_df, ktr_df,
        xte_df, yte_df, kte_df,
        output_dir, algo_list, seed,
        vectorizer, ngram_range, classifier_name)


    
def logistic_regression(
    xtr_df, ytr_df, ktr_df, 
    xte_df, yte_df, kte_df, 
    output_dir, algo_list, seed,
    vectorizer, ngram_range
):
    breakpoint()
    perf_metrics = set()
    # F1_SCORE = "f1_score"
    # ROC_AUC = "roc_auc"
    # PRECISION = "precision"
    # RECALL = "recall"
    # ACCURACY = "accuracy"
    
    # preproc_name = 'CountVectorizer'
    classifier_name = 'logistic_regression'
    # if using pre-trained vectorizer, pass it here
    pipe, clf_tuning_kws = make_clf_pipeline(
        vectorizer, classifier_name, memory_dir=None, vectorizer=None, ngram_range=ngram_range
        # vectorizer, classifier_name, memory_dir=output_dir, vectorizer=None, ngram_range=ngram_range
    )
    kfold_cv = sklearn.model_selection.StratifiedKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=seed
    )

    searcher = sklearn.model_selection.GridSearchCV(
        pipe,
        clf_tuning_kws,
        scoring=config.SCORING,
        cv=kfold_cv,
        return_train_score=True,
        refit=True,
        error_score="raise",
        verbose=3
    )
    # the grid search expects a column vector of labels
    x_vals = [str(doc) for doc in xtr_df.values[:,0]]
    y_vals = ytr_df.values[:, 0]
    searcher.fit(x_vals, y_vals)
    logging.info("Best parameters found: %s", searcher.best_params_)
    cv_tr_perf_df, cv_va_perf_df = extract_perf_df_from_cv_obj(searcher)
    out_csv_file = "perf_by_hyper_%dfoldCV_seed%d_{split}.csv" % (config.N_SPLITS, seed)
    out_csv_path = os.path.join(
        output_dir, out_csv_file
    )
    cv_tr_perf_df.to_csv(out_csv_path.format(split="train_cross_val"), index=False)
    cv_va_perf_df.to_csv(out_csv_path.format(split="test_cross_val"), index=False)
    logging.info("Wrote hyper search info to: %s", out_csv_path)

    save_fit_tokens(searcher.best_estimator_, output_dir)

    xte_vals = [str(doc) for doc in xte_df.values[:, 0]]
    yte_vals = yte_df.values[:, 0]
    yte_pred = searcher.best_estimator_.predict(xte_vals)
    yte_pred_proba = searcher.best_estimator_.predict_proba(xte_vals)[:, 1]

    conf_matrix = sklearn.metrics.confusion_matrix(yte_vals, yte_pred)
    logging.info("Confusion Matrix:\n%s", conf_matrix)  
    # @TODO: save confusion matrix to file
    # @TODO: save roc-auc to file

    # score with scoring value
    test_score = sklearn.metrics.get_scorer(config.SCORING)(
        searcher.best_estimator_, xte_vals, yte_vals
    )
    f1_score = sklearn.metrics.f1_score(yte_vals, yte_pred)
    roc_auc = sklearn.metrics.roc_auc_score(yte_vals, yte_pred_proba)
    precision = sklearn.metrics.precision_score(yte_vals, yte_pred)
    recall = sklearn.metrics.recall_score(yte_vals, yte_pred)
    accuracy = sklearn.metrics.accuracy_score(yte_vals, yte_pred)
    perf_metrics = {
        config.F1_SCORE: f1_score,
        config.ROC_AUC: roc_auc,
        config.PRECISION: precision,
        config.RECALL: recall,
        config.ACCURACY: accuracy
    }

    best_estimator_results = {
        "best_params": searcher.best_params_,
        "test_score": test_score,
    }

    best_estimator_df = pd.DataFrame([best_estimator_results])
    best_estimator_file_path = os.path.join(output_dir, "best_estimator_results.csv")
    best_estimator_df.to_csv(best_estimator_file_path, index=False)
    logging.info("Wrote best estimator results to: %s", best_estimator_file_path)
    logging.info("Test score: %f", test_score)

    return perf_metrics


def save_fit_tokens(best_pipeline, output_dir):
    """
    Saves the fitted tokens from the vectorizer in the pipeline to a JSON file.
    
    Parameters:
    best_pipeline (sklearn.pipeline.Pipeline): The trained pipeline containing the vectorizer.
    output_dir (str): The directory where the tokens will be saved.
    """
    vectorizer = best_pipeline.named_steps['txt2vec']
    tokens = vectorizer.get_feature_names_out()
    data_tools.save_csv_file(
        pd.DataFrame(tokens, columns=['tokens']),
        filename='fitted_tokens.csv',
        output_dir=output_dir
    )
    tokens_file_path = os.path.join(output_dir, "fitted_tokens.json")
    
    with open(tokens_file_path, 'w') as f:
        json.dump(tokens.tolist(), f)
    
    logging.info("Saved fitted tokens to: %s", tokens_file_path)


# TODO: implement ngram support
# TODO: implement vectorizer support
def make_clf_pipeline(
    preproc_name, clf_name, memory_dir, vectorizer=None, ngram_range=config.UNIGRAM):
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
    logging.info("Creating classifier pipeline with preproc: %s, clf: %s, vectorizer: %s, ngram_range: %s", preproc_name, clf_name, vectorizer, ngram_range)
    steps = list()

    # Use pre-trained vectorizer if provided, otherwise create new one
    if vectorizer is not None:
        logging.info("Using pre-trained vectorizer")
        steps.append(("txt2vec", "passthrough"))
    else:
        # Create new vectorizer with optional ngram_range 
        vectorizer_kwargs = {}
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
    pipe = sklearn.pipeline.Pipeline(steps, memory=memory_dir, verbose=True)
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
