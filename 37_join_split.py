import os
import pandas as pd

BIBLE_DIR = '../paralleltext/bibles/corpus'
ORIG_OUTPUT_DIR = 'output/KoplenigEtAl'
SPLIT_OUTPUT_DIR = 'output/KoplenigEtAlSpace'
EXCLUDED_BIBLE_FILE = 'bibles_to_exclude.txt'
SPECIAL_CHAR_FILE = 'bibles_with_non_standard_spaces.txt'
OUTPUT_DIR = SPLIT_OUTPUT_DIR


def get_list_of_bible_filenames():
    return os.listdir(BIBLE_DIR)


def get_excluded_bibles():
    with open(EXCLUDED_BIBLE_FILE) as f:
        lines = f.readlines()
    return [el.strip().split('#')[0].strip() for el in lines]


def get_special_char_bibles():
    with open(SPECIAL_CHAR_FILE) as f:
        lines = f.readlines()
    return [el.strip() for el in lines]


def get_df(file: str, adir: str) -> pd.DataFrame:
    df_p = pd.read_csv(os.path.join(adir, 'WordPasting', f'entropies_{file}.csv'))
    df_s = pd.read_csv(os.path.join(adir, 'WordSplitting', f'entropies_{file}.csv'))
    df_p['experiment'] = 'pasting'
    df_s['experiment'] = 'splitting'
    joined = pd.concat([df_p, df_s])
    joined['bible'] = file
    return joined


def get_df_from_split(file: str) -> pd.DataFrame:
    return get_df(file, SPLIT_OUTPUT_DIR)


def get_df_from_orig(file: str) -> pd.DataFrame:
    return get_df(file, ORIG_OUTPUT_DIR)


bibles = get_list_of_bible_filenames()
excluded_bibles = get_excluded_bibles()
special_char_bibles = get_special_char_bibles()
dfs = []
for bible in bibles:
    if bible in excluded_bibles:
        continue
    if bible in special_char_bibles:
        df = get_df_from_split(bible)
    else:
        df = get_df_from_orig(bible)
    dfs.append(df)

merged = pd.concat(dfs)
merged.to_csv(os.path.join(OUTPUT_DIR, 'merged.csv'))
