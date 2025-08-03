import glob
import os

import pandas as pd


def combine_results_csv():
    """
    Combines multiple CSV result files from a specified directory into a single CSV file.
    """
    # Define the directory where the result files are located.
    results_dir = "/Users/baba/proj/aphaproject/tufts_mph_ile/results/newres/"

    # List of the specific CSV files to be combined.
    files_to_combine = [
        "batch_run_results_part1.csv",
        "batch_run_results_part2.csv"
        # "naive_new_stopwords_batch_run_results.csv",
        # "rf_new_stopwords_batch_run_results.csv",
        # "xgboost_new_stopwords_no_cv_batch_run_results.csv",
        # "lr_new_stopwords_batch_run_results.csv",
    ]

    # Create a list to hold the dataframes
    df_list = []

    # Loop through the list of files, read each one into a dataframe, and append to the list
    for filename in files_to_combine:
        file_path = os.path.join(results_dir, filename)
        try:
            df = pd.read_csv(file_path)
            df_list.append(df)
            print(f"Successfully read {filename}")
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")

    # Check if any dataframes were loaded
    if not df_list:
        print("No dataframes were loaded. Exiting.")
        return

    # Concatenate all the dataframes in the list
    combined_df = pd.concat(df_list, ignore_index=True)

    # Define the output file path
    output_filename = "combined_results_20250803.csv"
    output_path = os.path.join(results_dir, output_filename)

    # Save the combined dataframe to a new CSV file
    combined_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully combined {len(df_list)} files into {output_path}")
    print(f"Total rows in combined file: {len(combined_df)}")


if __name__ == "__main__":
    combine_results_csv()
