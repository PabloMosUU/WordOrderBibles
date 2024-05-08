import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

def make_plot(entropies_dir: str, bible_filename: str) -> None:
    sub_dirs = {'wp': 'WordPasting', 'ws': 'WordSplitting'}
    wp_df, ws_df = [pd.read_csv(os.path.join(entropies_dir, sub_dir, f'entropies_{bible_filename}.csv')) for sub_dir in sub_dirs.values()]
    wp_df['experiment'] = 'pasting'
    ws_df['experiment'] = 'splitting'
    ws_df['iter_id'] = ws_df['iter_id'].apply(lambda x: -x)
    df = pd.concat([wp_df, ws_df])
    for lbl, grp in df.groupby('book'):
        xs = grp[grp['experiment'] == 'splitting']['D_order'].tolist()
        ys = grp[grp['experiment'] == 'splitting']['D_structure'].tolist()
        xp = grp[grp['experiment'] == 'pasting']['D_order'].tolist()
        yp = grp[grp['experiment'] == 'pasting']['D_structure'].tolist()
        labelss = grp[grp['experiment'] == 'splitting']['iter_id'].tolist()
        labelsp = grp[grp['experiment'] == 'pasting']['iter_id'].tolist()
        fig, ax = plt.subplots()
        ax.scatter(xs, ys)
        ax.scatter(xp, yp)
        plt.xlabel('Word order information')
        plt.ylabel('Word structure information')
        plt.title(f'{lbl}')
        for i, txt in enumerate(labelss):
            ax.annotate(txt, (xs[i], ys[i]), rotation=45)
        for i, txt in enumerate(labelsp):
            ax.annotate(txt, (xp[i], yp[i]), rotation=45)
        plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Required: entropies_dir bible_name')
    entropies_dir = sys.argv[1]
    bible_name = sys.argv[2]
    make_plot(entropies_dir, bible_name)