'''
split_dataset : 

Usage
-----
Defines two useful functions

* split_into_train_test

split_dataset(
    frac_test=0.5,
    random_state=42,
    perm_ids=
    raw_xlsx_file,
    xtext_colname='Narrative',
    y_colname='')

Calling save_split_on_disk will create a folder with two files

* train_data.csv.gz
* test_data.csv.gz

'''

import argparse
import copy
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd


def split_into_train_test(
        xdf,
        ydf,
        kdf,
        split_date='2022',
        date_column='Date',
        random_state=0):
    '''
    Splits the data into train and test sets based on a given date, preserving index and key order.

    Parameters:
    -----------
    xdf : pandas.DataFrame
        DataFrame containing features.
    ydf : pandas.DataFrame
        DataFrame containing labels.
    kdf : pandas.DataFrame
        DataFrame containing key information, including the date column.
    split_date : str
        Date string to split on. Can be 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'.
    date_column : str
        Name of the column containing dates in kdf.
    random_state : int
        Random state for reproducibility.

    Returns:
    --------
    tuple
        ((xtr, ytr, ktr), (xte, yte, kte))

    Example:
    --------
    >>> file_path = 'path/to/your/excel/file.xlsx'
    >>> xdf, ydf, kdf = read_excel_file(file_path)
    >>> (xtr, ytr, ktr), (xte, yte, kte) = split_into_train_test(xdf, ydf, kdf, split_date='2022-07-01', random_state=42)
    >>> print(f"Train size: {len(xtr)}, Test size: {len(xte)}")
    Train size: 181, Test size: 184
    '''
    assert xdf.index.equals(ydf.index) and xdf.index.equals(kdf.index), "xdf, ydf, and kdf must have the same index"
    assert date_column in kdf.columns, f"'{date_column}' column not found in kdf"

    # Convert split_date to datetime
    if len(split_date) == 4:  # YYYY
        split_datetime = datetime.strptime(split_date, '%Y')
    elif len(split_date) == 7:  # YYYY-MM
        split_datetime = datetime.strptime(split_date, '%Y-%m')
    elif len(split_date) == 10:  # YYYY-MM-DD
        split_datetime = datetime.strptime(split_date, '%Y-%m-%d')
    else:
        raise ValueError("split_date must be in 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD' format")

    # Ensure the date column is in datetime format
    kdf[date_column] = pd.to_datetime(kdf[date_column])

    # Split the data
    train_mask = kdf[date_column] < split_datetime
    test_mask = ~train_mask

    xtr_df, xte_df = xdf[train_mask], xdf[test_mask]
    ytr_df, yte_df = ydf[train_mask], ydf[test_mask]
    ktr_df, kte_df = kdf[train_mask], kdf[test_mask]

    # Create a shuffled index for each split, but keep the original index
    prng = np.random.default_rng(random_state)
    train_shuffle = prng.permutation(len(xtr_df))
    test_shuffle = prng.permutation(len(xte_df))

    # Apply the shuffle to the index, not the data itself
    xtr_df = xtr_df.iloc[train_shuffle]
    ytr_df = ytr_df.iloc[train_shuffle]
    ktr_df = ktr_df.iloc[train_shuffle]

    xte_df = xte_df.iloc[test_shuffle]
    yte_df = yte_df.iloc[test_shuffle]
    kte_df = kte_df.iloc[test_shuffle]

    logging.info(f"Train size: {len(xtr_df)}, Test size: {len(xte_df)}")
    logging.info(f'Train date range: {ktr_df[date_column].min()} to {ktr_df[date_column].max()}')
    logging.info(f'Test date range: {kte_df[date_column].min()} to {kte_df[date_column].max()}')

    return (xtr_df, ytr_df, ktr_df), (xte_df, yte_df, kte_df)
    

def read_xlsx_into_x_y_k_dfs(xlsx_path):
    """Read and clean the xlsx file into feature, label, and key dataframes"""
    xl_df = pd.read_excel(xlsx_path, sheet_name="Sheet1")

    clean_x_list = list()
    for row in range(xl_df.shape[0]):
        s = str(xl_df['Narrative'].values[row])
        clean_x_list.append(str.strip(s))
    xl_df['Narrative'] = clean_x_list

    key_cols = ['Date', 'Priority']
    keep_row_mask = np.logical_and(
        xl_df.Priority.isin([1,2,3,4,5]),
        xl_df.Narrative.str.len() >= 2)

    keep_df = xl_df.iloc[np.flatnonzero(keep_row_mask)][key_cols + ["Narrative"]].copy()
    keep_df['IsORI'] = keep_df['Priority'].isin([1,2,3]).astype(np.int32)
    x_df = keep_df[['Narrative']].copy()
    y_df = keep_df[['IsORI']].copy()    
    k_df = keep_df[key_cols].copy()

    for df in [x_df, y_df, k_df]:
        df.index.name = 'RowID_in_xlsx'
        df.index += 2 # from zero to one based, plus add one for header
    return x_df, y_df, k_df


def _make_simple_xdf_ydf():
    '''
    '''
    N = 100
    y_N = np.zeros(N, dtype=np.int32)
    y_N[-20:] = 1
    x_N2 = 0.01 * np.random.randn(N,2)
    x_N2[:,0] += y_N
    x_df = pd.DataFrame(x_N2, columns=['feat1', 'feat2'])
    y_df = pd.DataFrame(y_N[:,np.newaxis], columns=['label'])
    return x_df, y_df



if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx_path', type=str, required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--train_csv_filename_pattern', default='{varname}_train.csv')
    parser.add_argument('--test_csv_filename_pattern', default='{varname}_test.csv')
    parser.add_argument('--frac_test', required=True, type=float, default=0.2)
    parser.add_argument('--random_state', required=False, type=int, default=202407)
    parser.add_argument('--fold_id', required=False, type=int, default=0)
    args = parser.parse_args()

    x_df, y_df, k_df = read_xlsx_into_x_y_k_dfs(args.xlsx_path)
    (xtr_df, ytr_df, ktr_df), (xtest_df, ytest_df, ktest_df) = split_into_train_test(
        xdf=x_df,
        ydf=y_df,
        kdf=k_df,
        frac_test=args.frac_test,
        fold_id=args.fold_id,
        random_state=args.random_state)

    for [df, filename] in [[xtr_df, 'x_train.csv'], [ytr_df, 'y_train.csv'], [ktr_df, 'key_train.csv'],
						   [xtest_df, 'x_test.csv'], [ytest_df, 'y_test.csv'], [ktest_df, 'key_test.csv']]:
        save_csv_file(df, filename, output_dir=args.output_dir)
