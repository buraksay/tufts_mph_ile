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

    # --- 2. Configure Logging ---
    logging.basicConfig(
        format=config.LOG_MSG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        level=config.LOG_LEVEL,
    )

    # --- 3. Create a single batch directory for this entire run ---
    # The results for each individual task will be saved inside this directory.
    batch_output_dir = get_batch_dir(RES_DIR)
    logging.info(f"Master output directory for this batch run: {batch_output_dir}")

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
                run_single_task(
                    task_id=task_id,
                    task_tag=task_tag,
                    batch_output_dir=batch_output_dir,
                )
                logging.info(f"Finished task: {task_id} - {task_tag}")

    except FileNotFoundError:
        logging.error(f"Error: The file was not found at '{args.file}'")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logging.info("All tasks have been processed.")


if __name__ == "__main__":
    main()
