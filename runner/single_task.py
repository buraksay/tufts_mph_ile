import argparse
import logging
import os
import sys
from datetime import datetime

# s

# std-por-uni-tfidf-lr
# pun-alt-bi-bow-nb
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory, which is the project root (.../tufts_mph_ile)
CODE_DIR = os.path.dirname(script_dir)
# Add the project root to the Python path to ensure all modules can be found
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import config
import data_tools
from experiments import algo_mappings
from task_multiplexer import preprocess_data, split_data, train_classifier

DATA_DIR = os.environ.get(
    "DATA_DIR", config.DATA_DIR
)  # , os.path.join(CODE_DIR, "data"))
DATA_FILE = os.environ.get("DATA_FILE", config.DATA_FILE_NAME)
RES_DIR = os.environ.get("RES_DIR", os.path.join(CODE_DIR, config.RES_DIR_NAME))


# PREPROCESSOR = "tokenstem_tfidf"
# CLASSIFIER = "logistic_regression"

# OUTPUT_DIR = os.path.join(RES_DIR, f"analysis_seed{SEED}_spliton{SPLIT_DATE}")
# STEMMED_TOKENIZED_PATH = os.environ.get(
#     "STEMMED_TOKENIZED_PATH", os.path.join(OUTPUT_DIR, "x_stem_token.csv")
# )


def get_valid_algorithms():
    """Get all valid algorithm identifiers from algo_mappings."""
    valid_algorithms = set()
    for _, options in algo_mappings.items():
        for algo_key in options.keys():
            valid_algorithms.add(algo_key)
    return valid_algorithms


def validate_algo_list(algo_list, soft_errrors=False):
    valid_algorithms = get_valid_algorithms()
    missing_algorithms = []

    if(algo_list is None or len(algo_list) != 5):
        raise ValueError(f"The algorithm list must contain exactly 5 elements, but got {len(algo_list)} elements.")
    
    for algo in algo_list:
        if algo not in valid_algorithms:
            missing_algorithms.append(algo)

    is_valid = len(missing_algorithms) == 0

    if not is_valid and not soft_errrors:
        raise ValueError(f"The following algorithms were not found in experiments module: {missing_algorithms}")

    return is_valid, missing_algorithms


def run_single_task(task_id, task_tag, batch_output_dir):
    # Split the task string into individual algorithms
    algo_list = task_tag.split('-')
    is_valid, missing_algorithms = validate_algo_list(algo_list, soft_errrors=False)

    logging.info("Running task %s with tag %s", task_id, task_tag)

    OUTPUT_DIR = os.path.join(batch_output_dir, f"{task_id}_{task_tag}")
    os.makedirs(OUTPUT_DIR, exist_ok=False)
    logging.info("Output directory: %s", OUTPUT_DIR)

    xlsx_path = os.path.join(DATA_DIR, DATA_FILE)

    logging.info("Calling data_tools.read_xlsx_into_x_y_k_dfs")
    logging.info("INPUT xlsx_path: %s", xlsx_path)
    logging.info("OUTPUT dir: %s", OUTPUT_DIR)
    x_df, y_df, k_df = data_tools.read_xlsx_into_x_y_k_dfs(xlsx_path)
    logging.debug("X DataFrame head: %s", x_df.head())

    # DELETE BEGIN
    if False: #not False:
        import snippets
        snippets.run_frequency_analysis(x_df, OUTPUT_DIR)
    else:
    # END DELETE
        x_df_pp = preprocess_data(task_id, task_tag, output_dir=OUTPUT_DIR, algo_list=algo_list, df=x_df)

        (xtr_df, ytr_df, ktr_df), (xtest_df, ytest_df, ktest_df) = split_data(
            x_df=x_df_pp,
            y_df=y_df,
            k_df=k_df,
            split_date=config.SPLIT_DATE,
            seed=config.SEED,
            output_dir=OUTPUT_DIR,
        )
        logging.debug("Train DataFrame head: %s", xtr_df.head())
        train_classifier(
            xtr_df=xtr_df,
            ytr_df=ytr_df,
            ktr_df=ktr_df,
            xte_df=xtest_df,
            yte_df=ytest_df,
            kte_df=ktest_df,
            algo_list=algo_list,
            output_dir=OUTPUT_DIR,
            seed=config.SEED,
        )
        # train_classifier()
        # breakpoint()



def test_validate_algo_list():
    """Test the validate_algo_list function with a sample algorithm list."""
    algo_lists = [
        "sw.std-stem.por-ngram.uni-vec.tfidf-clf.nb",
        "sw.csw-stem.snow-ngram.bi-vec.bow-clf.lr",
        "sw.pun-stem.por-ngram.tri-vec.tfidf-clf.xgb",
        "sw.std-stem.snow-ngram.uni-vec.bow-clf.rf",
    ]

    for algo_list in algo_lists:
        is_valid, missing_algorithms = validate_algo_list(algo_list.split("-"), True)
        logging.info("%s is valid: %s", algo_list, is_valid)
        if not is_valid:
            logging.warning("Missing algorithms: %s", missing_algorithms)


if __name__ == '__main__':
    logging.basicConfig(
        format=config.LOG_MSG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        level=config.LOG_LEVEL,
    )
    parser = argparse.ArgumentParser(description="Run a single task with specified algorithms.")
    parser.add_argument("--id", type=str, required=True, help="Task ID")
    parser.add_argument("--tag", type=str, required=True, help="Task tag with algorithms")
    parser.add_argument("--batch", type=str, required=False, help="Batch output directory")
    args = parser.parse_args()
    task_id = args.id
    task_tag = args.tag
    batch_dir = args.batch 
    
    if not batch_dir:
        datetime_str = datetime.now().strftime("%Y%m%d%H%M")
        batch_dir = f"batch_{datetime_str}"
        
    BATCH_OUTPUT_DIR = os.path.join(RES_DIR, batch_dir)
    BATCH_OUTPUT_DIR = os.environ.get(
        "BATCH_OUTPUT_DIR", os.path.join(RES_DIR, batch_dir)
    )
    os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)
    logging.info(f"Batch output directory: {BATCH_OUTPUT_DIR}")
    logging.info(f"Code directory: {CODE_DIR}")
    logging.info(f"Data directory: {DATA_DIR}")
    logging.info(f"Data file: {DATA_FILE}")
    logging.info(f"Results directory: {RES_DIR}")
    logging.info(f"Seed: {config.SEED}")
    logging.info(f"Split date: {config.SPLIT_DATE}")

    # xlsx_path = os.path.join(DATA_DIR, "Data.xlsx")
    test_validate_algo_list()
    # run_single_task(task_id="EXP001", task_tag="sw.std-stem.por-ngram.uni-vec.tfidf-clf.lr", batch_output_dir=BATCH_OUTPUT_DIR)
    run_single_task(
        task_id=task_id, #"EXP001",
        task_tag=task_tag,#"sw.std-stem.snow-ngram.bi-vec.tfidf-clf.lr",
        batch_output_dir=BATCH_OUTPUT_DIR,
    )
