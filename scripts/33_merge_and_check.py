"""Merges the results of the pasting and splitting experiments.

Usage: python 33_merge_and_check.py [WORD_PASTING_DIRECTORY] [WORD_SPLITTING_DIRECTORY] [EXTENSION(csv)] [OUTPUT_FILENAME]
Dependencies: pandas
Author: Pablo Mosteiro
Status: Final
"""
import argparse
import pandas as pd
import os

def get_files(the_dir: str, ext: str) -> list:
    return [f for f in os.listdir(the_dir) if f.endswith(ext)]


def get_experiment_df(the_dir: str, ext: str, experiment: str) -> pd.DataFrame:
    filenames = get_files(the_dir, ext)
    filename_dataframe = {filename: pd.read_csv(os.path.join(the_dir, filename)) for filename in filenames}
    for filename, dataframe in filename_dataframe.items():
        dataframe['bible'] = filename.replace(ext, '').replace('entropies_',  '')
    exp_df = pd.concat(filename_dataframe.values())
    exp_df['experiment'] = experiment
    return exp_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('pasting_dir')
    parser.add_argument('splitting_dir')
    parser.add_argument('extension')
    parser.add_argument('output_filename')
    
    args = parser.parse_args()
    
    pasting_df = get_experiment_df(args.pasting_dir, args.extension, 'pasting')
    splitting_df = get_experiment_df(args.splitting_dir, args.extension, 'splitting')
    assert len(pasting_df.columns) == len(splitting_df.columns), f'\nPASTING: {pasting_df.columns}\nSPLITTING:{splitting_df.columns}\n'
    assert all([pasting_df.columns[i] == splitting_df.columns[i] for i in range(len(pasting_df.columns))])

    df = pd.concat([pasting_df, splitting_df])

    # Save the merged csv file.
    assert len(df[df.apply(lambda row: not row['bible'][3:11] == '-x-bible', 1)]) == 0
    df.to_csv(args.output_filename, index=False)
