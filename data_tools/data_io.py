import argparse
import logging
import os

import pandas as pd


def load_xdf_ydf_kdf_from_csv(
        csv_file_pattern="path/to/{varname}_train.csv",
        index_col="RowID_in_xlsx"
        ):
    df_by_name = {}
    for varname in ['x', 'y', 'key']:
         df = pd.read_csv(
            csv_file_pattern.format(varname=varname),
            index_col=index_col)
         df_by_name[varname] = df
    return df_by_name['x'], df_by_name['y'], df_by_name['key']

def save_csv_file(df, filename,
        output_dir,
        ):
    ''' Save a csv file, index assumed to be named elsewhere'''

    out_path = os.path.join(output_dir, filename)
    df.to_csv(out_path, header=True)
    logging.info("saved %s with shape %s to file %s" % (
        filename, df.shape, out_path))
    logging.info("[%s] sample 3 rows:%s", filename, df.head(3))


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file_pattern', default='{varname}_{split}.csv')
    args = parser.parse_args()

    for split in ['train', 'test']:
        logging.info("## %s SET", split.upper())
        csv_pat = args.csv_file_pattern.format(split=split, varname='{varname}')
        xdf, ydf, kdf = load_xdf_ydf_kdf_from_csv(csv_pat)

        logging.info("Y DataFrame describe: %s", ydf.describe())

        xdf['NarrativeLen'] = xdf['Narrative'].str.len()
        logging.info("X DataFrame NarrativeLen describe: %s", xdf[["NarrativeLen"]].describe())
