import argparse
import csv
import logging
import os
import sys

# --- Path Setup ---
# This ensures that modules from the project root (like config) and sibling directories can be imported.
# Get the directory of the current script (.../runner)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory, which is the project root (.../tufts_mph_ile)
CODE_DIR = os.path.dirname(script_dir)
# Add the project root to the Python path
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# --- Project Imports ---
# These can now be imported correctly because of the path setup above
import config
from runner.single_task import (DATA_DIR, DATA_FILE, RES_DIR, get_batch_dir,
                                run_single_task)


def main():
    """
    Parses an experiment matrix CSV file and runs each experiment as a single task.
    """
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Run a batch of experiments from a CSV file."
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Absolute path to the experiment matrix CSV file.",
    )
    args = parser.parse_args()

    # The results for each individual task will be saved inside this directory.
    batch_output_dir = get_batch_dir(RES_DIR)

    log_file_path = os.path.join(batch_output_dir, "batch_run.log")
    # Get the root logger. Don't use basicConfig, as it can only be called once.
    logger = logging.getLogger()
    logger.setLevel(config.LOG_LEVEL)  # Set the lowest level to capture all messages.
    # Create a formatter to use for both handlers
    formatter = logging.Formatter(
        fmt=config.LOG_MSG_FORMAT, datefmt=config.LOG_DATE_FORMAT
    )
    # Create a handler for console output (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # Create a handler for file output
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    results_file_path = os.path.join(batch_output_dir, "batch_run_results.csv")
    results_logger = logging.getLogger("results_logger")
    results_logger.setLevel(logging.INFO)
    # Add a handler to write to the results CSV file.
    results_file_handler = logging.FileHandler(results_file_path)
    # Use a minimal formatter that only outputs the message itself.
    results_file_handler.setFormatter(logging.Formatter("%(message)s"))
    results_logger.addHandler(results_file_handler)
    # Prevent results from being passed to the main logger.
    results_logger.propagate = False

    logging.info(f"Master output directory for this batch run: {batch_output_dir}")
    results_logger = logging.getLogger("results_logger")
    results_logger.setLevel(logging.INFO)
    logging.info(f"Logging results to: {results_file_path}")
    # Write the header to the results CSV file.
    # NOTE: Adjust the header based on the actual keys in the dictionary returned by run_single_task.
    results_header = "task_id,task_tag,f1_score,roc_auc,precision,recall,accuracy"
    results_logger.info(results_header)
    # --- 4. Read the CSV and run each task ---
    try:
        with open(args.file, mode="r", encoding="utf-8") as infile:
            # Use DictReader to easily access columns by name (e.g., 'ID', 'Tag')
            reader = csv.DictReader(infile)

            for row in reader:
                task_id = row.get("ID")
                task_tag = row.get("Tag")

                if not task_id or not task_tag:
                    logging.warning(f"Skipping row with missing ID or Tag: {row}")
                    continue

                logging.info("=" * 80)
                logging.info("=" * 80)
                logging.info(f"Starting task: ID='{task_id}', Tag='{task_tag}'")
                logging.info("=" * 80)
                logging.info("=" * 80)

                # Call the main function from single_task.py
                perf_metrics = run_single_task(
                    task_id=task_id,
                    task_tag=task_tag,
                    batch_output_dir=batch_output_dir,
                )
                logging.info(f"Finished task: {task_id} - {task_tag}")

                # Log the results to the dedicated CSV file.
                if perf_metrics:
                    f1 = perf_metrics.get(config.F1_SCORE, "N/A")
                    roc_auc = perf_metrics.get(config.ROC_AUC, "N/A")
                    precision = perf_metrics.get(config.PRECISION, "N/A")
                    recall = perf_metrics.get(config.RECALL, "N/A")
                    accuracy = perf_metrics.get(config.ACCURACY, "N/A")
                    results_logger.info(f"{task_id},{task_tag},{f1},{roc_auc},{precision},{recall},{accuracy}")
                else:
                    results_logger.info(f"{task_id},{task_tag},N/A,N/A,N/A,N/A,N/A")

    except FileNotFoundError:
        logging.error(f"Error: The file was not found at '{args.file}'")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logging.info("All tasks have been processed.")


if __name__ == "__main__":
    main()
