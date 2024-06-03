import os
import pandas as pd
import sys

from util import make_plot_


def make_plot(entropies_dir: str, bible_filename: str, output_dir: str) -> None:
    sub_dirs = {'wp': 'WordPasting', 'ws': 'WordSplitting'}
    wp_df, ws_df = [pd.read_csv(os.path.join(entropies_dir, sub_dir, f'entropies_{bible_filename}.csv'))
                    for sub_dir in sub_dirs.values()]
    wp_df['experiment'] = 'pasting'
    ws_df['experiment'] = 'splitting'
    ws_df['iter_id'] = ws_df['iter_id'].apply(lambda x: -x)
    df = pd.concat([wp_df, ws_df])
    make_plot_(df, bible_filename, output_dir)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        raise ValueError('Required: entropies_dir bible_name output_dir')
    entropies_parent_dir = sys.argv[1]
    bible_name = sys.argv[2]
    output_image_dir = sys.argv[3]
    make_plot(entropies_parent_dir, bible_name, output_image_dir)
