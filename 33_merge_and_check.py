import pandas as pd
import os

PASTING_DIR = 'output/KoplenigEtAl/WordPasting/'
SPLITTING_DIR = 'output/KoplenigEtAl/WordSplitting/'
EXTENSION = '.csv'
OUTPUT_FILENAME = 'output/KoplenigEtAl/merged.csv'


def get_files(the_dir: str, ext: str) -> list:
    return [f for f in os.listdir(the_dir) if f.endswith(ext)]


def get_experiment_df(the_dir: str, ext: str, experiment: str) -> pd.DataFrame:
    filenames = get_files(the_dir, ext)
    filename_dataframe = {filename: pd.read_csv(os.path.join(the_dir, filename)) for filename in filenames}
    for filename, dataframe in filename_dataframe.items():
        dataframe['bible'] = filename.replace(EXTENSION, '').replace('entropies_',  '')
    exp_df = pd.concat(filename_dataframe.values())
    exp_df['experiment'] = experiment
    return exp_df


pasting_df = get_experiment_df(PASTING_DIR, EXTENSION, 'pasting')
splitting_df = get_experiment_df(SPLITTING_DIR, EXTENSION, 'splitting')
assert len(pasting_df.columns) == len(splitting_df.columns)
assert all([pasting_df.columns[i] == splitting_df.columns[i] for i in range(len(pasting_df.columns))])

df = pd.concat([pasting_df, splitting_df])

# Save the merged csv file.
assert len(df[df.apply(lambda row: not row['bible'][3:11] == '-x-bible', 1)]) == 0
df.to_csv(OUTPUT_FILENAME, index=False)
