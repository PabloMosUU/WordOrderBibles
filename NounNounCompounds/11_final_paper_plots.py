from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from scipy.stats import permutation_test

sys.path.append('..')
import data

ENTROPIES_FILENAME = '../output/KoplenigEtAl/merged.csv'
SEL_LANGS = ('eng', 'deu', 'nld')
BIBLE_LOCATION = '1_relevant_bibles'
BOOKS = [40, 41, 42, 43, 44, 66]
lang_color = {'eng': 'b', 'nld': 'r', 'deu': 'g'}
MIN_VERSE_FRAC = 0.9


def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def histogram_deltas(book_deltas: dict[str:dict[str:float]]) -> None:
    # Histogram the deltas to see if they are symmetric around a central value (assumption for Wilcoxon's test)
    for axis in book_deltas['Matthew'].keys():
        plt.clf()
        plt.hist([deltas[axis] for deltas in book_deltas.values()])
        plt.title(f'delta_D_{axis}')
        plt.show()


def check_student(book_values: dict[str:dict[str:list]]) -> None:
    for axis in list(book_values.values())[0].keys():
        plt.clf()
        no_merges = [axis_values[axis][0] for book, axis_values in book_values.items()]
        all_merges = [axis_values[axis][1] for book, axis_values in book_values.items()]
        no_merges_var = np.var(no_merges)
        all_merges_var = np.var(all_merges)
        print(f'{axis} variances: {no_merges_var}, {all_merges_var}')


def run_permutation_test(book_values: dict[str:dict[str:list]]) -> None:
    # The hypothesis is delta_D_order < 0; delta_D_structure > 0
    for axis in ('order', 'structure'):
        # x is before merging; y is after merging
        x = [axis_values[axis][0] for book, axis_values in book_values.items()]
        y = [axis_values[axis][1] for book, axis_values in book_values.items()]
        less_or_greater = 'less' if axis == 'structure' else 'greater'
        inverse_alt = '>' if less_or_greater == 'less' else '<'
        res = permutation_test((x, y), statistic, vectorized=True,
                               n_resamples=np.inf, alternative=less_or_greater)
        print(f'Prob(delta_D_{axis} {inverse_alt}= {-res.statistic}) under the null hypothesis is {100 * res.pvalue:.1f}%.')

def get_verse_len_df(lang: str, previously_excluded: list, book_id_name: pd.DataFrame) -> pd.DataFrame:
    files, books, verses = [], [], []
    for file in os.listdir(BIBLE_LOCATION):
        if not file.startswith(lang):
            continue
        if file in previously_excluded:
            continue
        # noinspection PyUnresolvedReferences
        bible = data.parse_file(os.path.join(BIBLE_LOCATION, file), 'pbc').join_by_toc()
        for book_id in BOOKS:
            book = book_id_name[book_id_name['book_id'] == book_id]['book'].tolist()[0]
            n_verses = len(bible[2][book_id])
            files.append(file)
            books.append(book)
            verses.append(n_verses)
    verse_df = pd.DataFrame({'file': files, 'book': books, 'n_verses': verses})
    book_max_verses = verse_df[['book', 'n_verses']].groupby('book').max().reset_index(drop=False)
    book_max_verses.rename(columns={'n_verses': 'max_verses'}, inplace=True)
    verse_df = verse_df.merge(book_max_verses, on='book', how='left').reset_index(drop=True)
    return verse_df

def single_axis_mean_individual_comparison(book_df: pd.DataFrame, axis: str) -> None:
    assert book_df['book'].nunique() == 1
    book_name = list(book_df['book'].unique())[0]
    n_merge_quantities = book_df[['merged', 'D_order', 'D_structure']].groupby('merged').mean().reset_index(
        drop=False
    )
    mean_values = n_merge_quantities[axis].tolist()
    plt.clf()
    plt.hist(book_df[book_df['merged'] == 0][axis], label='Individual translations, no merges')
    plt.axvline(x=mean_values[1], color='red', linestyle='--', label='Average over translations, all merges')
    plt.title(f'No-merge {axis} for book {book_name}')
    plt.legend()
    plt.show()

def compare_mean_merged_with_individual_non_merged(book_df: pd.DataFrame) -> None:
    # See where the average merged quantity lies in the histogram of no-merge quantities
    single_axis_mean_individual_comparison(book_df, 'D_order')
    single_axis_mean_individual_comparison(book_df, 'D_structure')

# noinspection PyPep8Naming
def produce_results(nn_pastes_dir: str, output_fig_dir: str) -> None:
    # ### Old data: select data points with 0 pastes in the chosen languages only. Also pastes in English
    df = pd.read_csv(ENTROPIES_FILENAME)
    df['language'] = df['bible'].apply(lambda x: x[:3])
    df = df[df['language'].apply(lambda x: x in SEL_LANGS)].reset_index(drop=True)
    df = df[(df['iter_id'] == 0) | (df['language'] == 'eng')].reset_index(drop=True)
    df['bible_id'] = df['bible'].apply(lambda x: x.strip().replace('.txt', '').replace('-bible', ''))
    df_sel_langs = df[df['experiment'] == 'pasting'].reset_index(drop=True)

    # ### New data
    new_files = [file for file in os.listdir(nn_pastes_dir)
                 if file.startswith('eng-x-bible-') and file.endswith('.txt.csv')]
    all_df = []
    for i, file in enumerate(new_files):
        df = pd.read_csv(os.path.join(nn_pastes_dir, file))
        if len(df) == 0:
            continue
        df.fillna({'merged_pair': ''}, inplace=True)
        df['filename'] = file
        all_df.append(df)
    merged_df = pd.concat(all_df)

    # There should be one entropy for each book, for each number of merges, and for each filename
    # (the merged_pair is associated 1-1 with n_merges)
    assert all([len(grp) == 3 for _, grp in merged_df.groupby(['book_id', 'n_merges', 'merged_pair', 'filename'])])

    # Convert the new data into a data frame that is similar to the old data
    book_ids, n_merges, merged_pair, filename, H_orig, H_order, H_structure, D_order, D_structure = ([], [], [], [], [],
                                                                                                     [], [], [], [])
    for lbl, grp in merged_df.groupby(['book_id', 'n_merges', 'merged_pair', 'filename']):
        H = {key: grp[grp['text_version'] == key]['entropy'].tolist()[0] for key in grp['text_version'].unique()}
        # noinspection PyUnresolvedReferences
        book_ids.append(lbl[0])
        # noinspection PyUnresolvedReferences
        n_merges.append(lbl[1])
        # noinspection PyUnresolvedReferences
        merged_pair.append(lbl[2])
        # noinspection PyUnresolvedReferences
        filename.append(lbl[3])
        H_orig.append(H['orig'])
        H_order.append(H['shuffled'])
        H_structure.append(H['masked'])
        D_order.append(H['shuffled'] - H['orig'])
        D_structure.append(H['masked'] - H['orig'])
    joined_df = pd.DataFrame({'book_id': book_ids, 'n_merges': n_merges, 'merged_pair': merged_pair,
                              'filename': filename, 'H_orig': H_orig, 'H_order': H_order, 'H_structure': H_structure,
                              'D_order': D_order, 'D_structure': D_structure})
    joined_df['filename'] = joined_df['filename'].apply(lambda x: x.strip().replace('.csv', ''))
    assert len(joined_df) == len(merged_df) / 3

    # Compatibility check
    # See that the values at 0 merges match between the old and new data
    old_unpasted = df_sel_langs[(df_sel_langs['iter_id'] == 0) &
                                (df_sel_langs['bible_id'].apply(lambda x: x.startswith('eng')))].reset_index(drop=True)
    new_unpasted = joined_df[joined_df['n_merges'] == 0].reset_index(drop=True)
    assert len(old_unpasted) == len(new_unpasted)
    old_unpasted['bible_book'] = old_unpasted.apply(lambda row: f'{row["bible"]}_{row["book_id"]}', 1)
    new_unpasted['bible_book'] = new_unpasted.apply(lambda row: f'{row["filename"]}_{row["book_id"]}', 1)
    assert len(old_unpasted) == old_unpasted['bible_book'].nunique()
    assert len(new_unpasted) == new_unpasted['bible_book'].nunique()
    old_and_new = old_unpasted.merge(new_unpasted, on='bible_book', how='inner', validate='1:1')
    assert len(old_and_new) == len(old_unpasted)
    key_map = {'orig': 'H_orig', 'shuffled': 'H_order', 'masked': 'H_structure'}
    for key, val in key_map.items():
        old_and_new[f'{val}_diff'] = old_and_new.apply(lambda row: abs(row[key] - row[val]), 1)
        old_and_new[f'{val}_mean'] = old_and_new.apply(lambda row: 0.5 * (row[key] + row[val]), 1)
        old_and_new[f'{val}_fracdiff'] = old_and_new.apply(lambda row: row[f'{val}_diff'] / row[f'{val}_mean'], 1)
    all_frac_diffs = []
    for val in key_map.values():
        all_frac_diffs += old_and_new[f'{val}_fracdiff'].tolist()
    assert len(all_frac_diffs) == 3 * len(old_and_new)
    largest_percentual_difference = max(all_frac_diffs)*100
    assert largest_percentual_difference < 5

    # ## Exclude old texts
    # Details in notebook 8
    # Bibles without a year for which we found more information
    excluded_bibles = ['deu-x-bible-bolsinger.txt']
    df.loc[df['filename'] == 'eng-x-bible-treeoflife.txt', ['year_short', 'year_long']] = 2009
    df.loc[df['filename'] == 'deu-x-bible-erben.txt', ['year_short', 'year_long']] = 1739
    # Bibles with known years before 1800
    excluded_bibles += ['deu-x-bible-luther1545letztehand.txt', 'deu-x-bible-erben.txt']
    excluded_bibles += ['nld-x-bible-statenvertaling.txt']
    excluded_bibles += ['eng-x-bible-kingjames.txt']

    # Exclude all bibles that contain fewer than MIN_VERSE_FRAC of the maximum number of verses for at least one book.
    book_id_map = df_sel_langs[['book_id', 'book']].drop_duplicates()
    verse_len_dfs = {lang: get_verse_len_df(lang, excluded_bibles, book_id_map) for lang in lang_color.keys()}
    for lang in lang_color.keys():
        vdf = verse_len_dfs[lang]
        to_remove = vdf[vdf['n_verses'] < vdf['max_verses'] * MIN_VERSE_FRAC]['file'].drop_duplicates().tolist()
        excluded_bibles += to_remove

    # remove the excluded_bibles from the datasets
    old_data = df_sel_langs[df_sel_langs['bible'].apply(lambda x: x not in excluded_bibles)].reset_index(drop=True)
    assert len(old_data) < len(df_sel_langs)
    new_data = joined_df[joined_df['filename'].apply(lambda x: x not in excluded_bibles)].reset_index(drop=True)
    assert len(new_data) < len(joined_df)

    # add the best-fit lines produced by Koplenig et al.
    fit_params = pd.read_csv('9_koplenig_et_al_fit_params.csv', sep=';')
    fit_params['beta_0'] = fit_params['beta_0'].apply(lambda x: float(x.replace(',', '.')))
    fit_params['beta_1'] = fit_params['beta_1'].apply(lambda x: float(x.replace(',', '.')))

    # Add book name to new dataset
    for lbl, grp in new_data.groupby(['book_id', 'n_merges']):
        assert len(grp) == grp['filename'].nunique()
    new_data_book = new_data.merge(book_id_map, on='book_id', how='left')
    assert len(new_data_book) == len(new_data)
    new_data_long = new_data_book[new_data_book['filename'].apply(lambda x: x not in excluded_bibles)].reset_index(
        drop=True
    )

    """
    # Drop book-translations that contain fewer than some fraction of the maximum number of NN pairs for that book
    book_max_merges = {lbl: grp['n_merges'].max() * 0.95
                       for lbl, grp in new_data_long[['book', 'n_merges']].groupby('book')}
    excluded_book_translations = []
    for lbl, grp in new_data_long.groupby('book'):
        for translation, n_merges_df in grp.groupby('filename'):
            if n_merges_df['n_merges'].max() < book_max_merges[lbl]:
                excluded_book_translations.append((lbl, translation))
    new_data_long = new_data_long[new_data_long.apply(lambda row: not any([row['book'] == el[0]
                                                                           and row['filename'] == el[1]
                                                                           for el in excluded_book_translations]),
                                                      1)].reset_index(drop=True)
    # Drop all merges that are above some fraction of the maximum number of noun-noun merges for that book
    new_data_long = new_data_long[new_data_long.apply(lambda row: any([row['book'] == book
                                                                       and row['n_merges'] <= max_nn_merges
                                            for book, max_nn_merges in book_max_merges.items()]),
                                                      1)].reset_index(drop=True)
    """

    book_values = {}
    for book_name in old_data['book'].unique():
        book_df = old_data[(old_data['book'] == book_name) & (old_data['iter_id'] == 0)].reset_index(drop=True)
        assert len(book_df) == book_df['bible_id'].nunique(), \
            (book_name, str(len(book_df)), str(book_df['bible_id'].nunique()))
        fig, ax = plt.subplots()
        for lang, point_color in lang_color.items():
            if lang == 'eng':
                continue
            lang_df = book_df[book_df['language'] == lang].reset_index(drop=True)
            x = lang_df['D_order'].tolist()
            y = lang_df['D_structure'].tolist()
            mean_x, mean_y = [[np.mean(el)] for el in (x, y)]
            ax.scatter(x=mean_x, y=mean_y, c=point_color, label=lang)
            ax.annotate(lang, (mean_x[0], mean_y[0]), rotation=45)

        # Plot the new pastes
        book_df = new_data_long[new_data_long['book'] == book_name].reset_index(drop=True)
        book_df_grouped = book_df[['n_merges', 'filename']].groupby('filename')
        assert all([len(grp) == grp['n_merges'].nunique() for _, grp in book_df_grouped])
        filename_max_n_merges = book_df_grouped.max().reset_index().rename(columns={'n_merges': 'max_n_merges'})
        book_df = book_df.merge(filename_max_n_merges, on='filename', how='left')
        book_df['merged'] = book_df.apply(
            lambda row: 1 if row['n_merges'] == row['max_n_merges'] else (0 if row['n_merges'] == 0 else -1), 1
        )
        book_df = book_df[book_df['merged'] != -1].reset_index(drop=True)
        n_merge_quantities = book_df[['merged', 'D_order', 'D_structure']].groupby('merged').mean().reset_index(
            drop=False
        )
        x = n_merge_quantities['D_order'].tolist()
        y = n_merge_quantities['D_structure'].tolist()
        n_merge_quantities['label'] = n_merge_quantities['merged'].map({0: 'eng-orig', 1: 'eng-nn-pasted'})
        labels = n_merge_quantities['label'].tolist()
        ax.scatter(x, y, c='blue')
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]), rotation=45)
        # Save the deltas associated with nn pasting
        assert len(x) == 2 and len(y) == 2
        book_values[book_name] = {'order': x, 'structure': y}

        # compare_mean_merged_with_individual_non_merged(book_df)
        # This was not very illuminating because we are interested in paired differences

        # Plot the old pastes data
        old_merges = old_data[old_data.apply(
            lambda row: row['book'] == book_name and 0 < row['iter_id'] <= 200 and row['language'] == 'eng', 1
        )]
        n_merge_quantities = old_merges[['iter_id', 'D_order', 'D_structure']].groupby('iter_id').mean().reset_index(
            drop=False
        )
        x = n_merge_quantities['D_order'].tolist()
        y = n_merge_quantities['D_structure'].tolist()
        ax.scatter(x, y, label='any word pair pastes', c='cyan')
        labels = n_merge_quantities['iter_id'].tolist()
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]), rotation=45)

        # Plot the fit line from Koplenig et al
        fit_x = np.arange(book_df['D_order'].min(), book_df['D_order'].max(),
                          (book_df['D_order'].max() - book_df['D_order'].min()) / 100)
        beta_0 = fit_params[fit_params['book'] == book_name]['beta_0'].tolist()[0]
        beta_1 = fit_params[fit_params['book'] == book_name]['beta_1'].tolist()[0]
        fit_y = [beta_0 + beta_1 / el for el in fit_x]
        ax.plot(fit_x, fit_y, linestyle='dashed', label='Koplenig et al fit line')

        plt.xlabel('Word order information')
        plt.ylabel('Word structure information')
        plt.title(book_name)
        plot_filename = f'{output_fig_dir}/nn_pastes_{book_name}.png'
        print(f'INFO: saving {plot_filename}')
        plt.savefig(plot_filename)

    language_bibles = defaultdict(list)
    for bible in [el for el in old_data['bible'].unique() if el not in excluded_bibles]:
        language_bibles[bible[:3]].append(bible)
    print('Bible counts:', {key: len(val) for key, val in language_bibles.items()})

    # Check the maximum number of merges for each translation
    book_max_verses = {book: max_n_verses for book, max_n_verses in verse_len_dfs['eng'][['book', 'max_verses']].values}
    for book, filename_n_merges in new_data_long[['filename', 'book', 'n_merges']].groupby('book'):
        assert all([len(grp) == grp['n_merges'].nunique() for lbl, grp in filename_n_merges.groupby('filename')])
        max_n_merges = [n_merges['n_merges'].max() for _, n_merges in filename_n_merges.groupby('filename')]
        print(f'{book} & {book_max_verses[book]} & {np.mean(max_n_merges):.1f} \\\\')

    # List of bibles used
    new_bibles = new_data_long[new_data_long['filename'].apply(lambda x: x not in excluded_bibles)]['filename'].unique()
    old_bibles = sorted(old_data[old_data['bible'].apply(lambda x: x not in excluded_bibles)]['bible'].unique())
    assert len([el for el in new_bibles if el not in old_bibles]) == 0
    assert all([not el.startswith('eng') for el in old_bibles if el not in new_bibles])
    assert len(old_bibles) == sum([len(el) for el in language_bibles.values()])
    for i in range(int(len(old_bibles) / 2)):
        print(" & ".join(f'\\texttt{"{"}{el}{"}"}' for el in (old_bibles[2*i], old_bibles[2*i+1])) + ' \\\\')

    # Check assumption for Wilcoxon's test
    """
    book_deltas = {book: {axis: el[1] - el[0] for axis, el in axis_values.items()} for book, axis_values in book_values.items()}
    histogram_deltas(book_deltas)
    """

    # Check assumption for Student's t-test
    # check_student(book_values)

    # Permutation test
    run_permutation_test(book_values)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError(f'Input: {sys.argv[0]} <nn_paste_dir> <output_fig_dir>')
    nn_paste_dir = sys.argv[1]
    output_fig_dir = sys.argv[2]
    produce_results(nn_paste_dir, output_fig_dir)
