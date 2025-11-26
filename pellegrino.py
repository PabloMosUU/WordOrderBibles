import sys
import pandas as pd
import os
from create_full_information_csv import bookdict
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

SELECTED_LANGUAGES = ["chr", "cmn", "deu", "eng", "esk", "grc", "vie", "xuo", "zul", "tam", "qvw"]
BIBLES_TO_EXCLUDE = 'bibles_to_exclude.txt'

def get_lang_curve(df, lang, avg_lang, smooth_curves=False):
    # This is for a given book_id
    lang_df = df[df['language'] == lang].reset_index(drop=True)

    if len(lang_df) == 0:
        print(f'WARNING: skipping {lang}')
        return [], []

    # Average over translations in a given language or select one translation
    if avg_lang:
        reduced_df = lang_df[['signed_iter_id', 'D_order', 'D_structure']].groupby('signed_iter_id').mean().reset_index(drop=True)
    else:
        bible = lang_df['bible'].tolist()[0]
        reduced_df = lang_df[lang_df['bible'] == bible].reset_index(drop=True)
    reduced_df.sort_values(by='D_order', ascending=True, inplace=True)

    x = reduced_df['D_order'].tolist()
    y = reduced_df['D_structure'].tolist()

    if smooth_curves:
        # Apply smoothing (window_length must be odd)
        window_length = 31
        polyorder = 3

        y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder)
        return x, y_smooth

    return x, y

def get_real_curve(full_df: pd.DataFrame, avg_lang: bool, book_id: int, bad_bibles: list[str]) -> list[list]:
    # For now we only accept averaging over translations in a language
    assert avg_lang
    # Choose only iter_id = 0 for pasting experiments
    zero_df = full_df[(full_df['iter_id'] == 0) & (full_df['experiment'] == 'pasting')].reset_index(drop=True)
    # Choose only the book that we are interested in
    books_df = zero_df[zero_df['book_id'].apply(lambda x: x == book_id)].reset_index(drop=True)
    # Choose only the languages that we are interested in
    lang_df = books_df[books_df['bible'].apply(lambda x: any([x.startswith(el) for el in SELECTED_LANGUAGES]))].reset_index(drop=True)
    # Exclude the bad bibles
    good_bible_df = lang_df[lang_df['bible'].apply(lambda x: x not in bad_bibles)].reset_index(drop=True)
    # Now we should have a single row per bible
    assert all([len(grp) == 1 for _, grp in good_bible_df.groupby('bible')])
    # Create a column for the language
    good_bible_df['language'] = good_bible_df['bible'].apply(lambda x: x.split('-')[0])
    # Average values over translations in the same language
    outcome_df = good_bible_df[['D_order', 'D_structure', 'language']].groupby('language').mean().reset_index(drop=False)

    return [outcome_df[col].tolist() for col in ('language', 'D_order', 'D_structure')]


def run_book(df: pd.DataFrame, avg_lang: bool, book_name: str, bad_bibles: list[str], overlay_curves: bool,
             select_language=None, smooth_curves=False) -> None:
    assert set(df['experiment'].unique()) == {'pasting', 'splitting'}
    df['signed_iter_id'] = df.apply(lambda row: row['iter_id'] if row['experiment'] == 'splitting' else -row['iter_id'], 1)

    # Now for a given book_id (given) and bible the signed_iter_ids should be unique, except for iter_id=0, which appears twice
    for lbl, grp in df.groupby('bible'):
        assert len(grp) == grp['signed_iter_id'].nunique() + 1

    # Iterate over languages and extract the curves
    curves = {}
    for lang in SELECTED_LANGUAGES:
        if select_language and lang != select_language:
            continue
        x, y = get_lang_curve(df, lang, avg_lang, smooth_curves)
        curves[lang] = (x, y)

    # Extract the curve that corresponds to 0 splits/pastes
    labels_real, x_real, y_real = get_real_curve(df, avg_lang, {v: k for k, v in bookdict.items()}[book_name], bad_bibles)

    # Desired display ranges
    x_min, x_max = 0, max(x_real)*4
    y_min, y_max = 0, max(y_real)*4

    # Plot the 0-split-0-paste curve as markers only
    colors = {lang: [
        "red", "blue", "green", "orange", "purple",
        "brown", "cyan", "magenta", "olive", "teal", "gold"
    ][i] for i, lang in enumerate(SELECTED_LANGUAGES)}
    plt.figure(figsize=(8, 6))
    for i, x_val in enumerate(x_real):
        plt.scatter(x_val, y_real[i], color=colors[labels_real[i]])

    # Add labels with 45-degree rotation
    for xi, yi, label in zip(x_real, y_real, labels_real):
        plt.text(xi, yi, label, rotation=45, fontsize=10, ha='left', va='bottom')

    if overlay_curves:
        # Plot each of the splitting-and-pasting curves on top of those
        for lang in curves.keys():
            plt.plot(curves[lang][0], curves[lang][1], label=lang, color=colors[lang])  # no marker, just line

        # Apply axis limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # Legend
        plt.legend(loc="upper right")

    # Optional: add axis labels and title
    plt.xlabel('D_order')
    plt.ylabel('D_structure')
    plt.title(f'{book_name}')
    plt.tight_layout()
    plt.show()

def exclude_bad_bibles(df: pd.DataFrame) -> tuple[pd.DataFrame,list]:
    with open(BIBLES_TO_EXCLUDE) as f:
        lines = set([el.strip() for el in f.readlines()])
    df = df[df['bible'].apply(lambda x: x.strip() not in lines)].reset_index(drop=True)
    return df, list(lines)

def run_pipeline(df: pd.DataFrame, avg_lang: bool) -> None:

    # Exclude corrupted bibles
    df, bad_bibles = exclude_bad_bibles(df)

    # Select the data from the df
    df = df[df['bible'].apply(lambda x: any([x.startswith(el) for el in SELECTED_LANGUAGES]))].reset_index(drop=True)

    df['language'] = df['bible'].apply(lambda x: x.split('-')[0])

    # Run the following program once per book
    for book_id, book_name in bookdict.items():
        run_book(df[df['book_id'] == book_id].reset_index(drop=True), avg_lang, book_name, bad_bibles)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError(f'ERROR: usage {sys.argv[0]} <directory> <avg_lang>')
    directory = sys.argv[1]
    avg_lang = bool(sys.argv[2])
    df = pd.read_csv(os.path.join(directory, 'merged.csv'))
    run_pipeline(df, avg_lang)
