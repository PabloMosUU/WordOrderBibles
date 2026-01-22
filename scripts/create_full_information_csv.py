import pandas as pd 
import matplotlib.pyplot as plt
import os
import json
import sys
from matplotlib.backends.backend_pdf import PdfPages

bookdict = {40: 'Matthew', 41:'Mark', 42: "Luke", 43: "John", 44:"Acts", 66:"Revelation"}

def read_csv_file(filename: str) -> pd.DataFrame:
    """
    Read csv file and add bible language
    """
    df = pd.read_csv(filename)
    df['bible_language'] = df['bible'].str[:3] # add bible language
    return df


def create_filename_book_verse_dict(bible_directory: str, df: pd.DataFrame):
    """
    Create dictionaries with info about the amount of verses, words and chars
    """
    verses_dict = {}
    char_dict = {}
    word_dict = {}
    max_verses_dict = {}

    unique_files = df['bible'].unique()

    for filename in unique_files:
        file_path = os.path.join(bible_directory, filename)
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.readlines()
                    subdict_verses = {}
                    subdict_char = {}
                    subdict_words = {}

                    for biblebook in [40, 41, 42, 43, 44, 66]:
                        count = 0
                        charcount = 0
                        word_count = 0

                        for line in content:
                            if line.startswith(str(biblebook)):
                                count += 1
                                charcount += len(line)
                                word_count += len(line.split(" "))

                        subdict_verses[biblebook] = count
                        subdict_char[biblebook] = charcount
                        subdict_words[biblebook] = word_count
                        max_verses_dict[biblebook] = max(count, max_verses_dict.get(biblebook, 0))

                verses_dict[filename] = subdict_verses
                char_dict[filename] = subdict_char
                word_dict[filename] = subdict_words

            except Exception as e:
                continue
        else:
            continue

    return verses_dict, char_dict, max_verses_dict, word_dict


def plot_bible_lengths(length_dict: dict, save_filepath: str):
    """
    Plot the number of verses in the different bibles in a histogram
    """
    output_pdf_path = os.path.join(save_filepath, f"Verse_distribution.pdf")
    with PdfPages(output_pdf_path) as pdf:
        for book in bookdict.keys():
            verselist = []
            for _, books in length_dict.items():

                for biblebook, verses in books.items():
                    if biblebook == book:
                        verselist.append(verses)

            plt.hist(verselist, bins=70, color='skyblue', edgecolor='black')
            plt.xlabel('Bible length')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of verse length for the bible book {bookdict[book]}')
            pdf.savefig()
            plt.close()


def get_value_from_dict(row, dict):
    """
    Helps to get certain values from a dictionary
    """
    return dict.get(row['bible'], {}).get(row['book_id'], None)


def add_verse_bit_info(bible_directory: str, df: pd.DataFrame):
    """
    Add info about verse length, bits per verse & bits per word
    """
    verse_dict, char_dict, max_verses, word_dict = create_filename_book_verse_dict(bible_directory, df)
    df['verse_length'] = df.apply(lambda row: get_value_from_dict(row, verse_dict), axis=1)
    df['char_length'] = df.apply(lambda row: get_value_from_dict(row, char_dict), axis=1)
    df['word_length'] = df.apply(lambda row: get_value_from_dict(row, word_dict), axis=1)

    df.dropna(subset=['verse_length'], inplace=True)
    df.dropna(subset=['char_length'], inplace=True)
    df.dropna(subset=['word_length'], inplace=True)

    df['D_structure_verse'] = (df['D_structure'] * df['char_length']) / df['verse_length'] 
    df['D_order_verse'] = (df['D_order'] * df['char_length']) / df['verse_length'] 

    df['D_structure_word'] = (df['D_structure'] * df['char_length']) / df['word_length'] 
    df['D_order_word'] = (df['D_order'] * df['char_length']) / df['word_length'] 

    return df, max_verses, verse_dict


def create_full_info_file(bible_directory: str, filename: str):
    """
    Create csv file with added columns for bpw and bpv, and dictionary of verse lengths
    """
    df = read_csv_file(filename)
    all_info_df, max_verses, verse_dict = add_verse_bit_info(bible_directory, df)

    all_info_df.to_csv("full_info.csv", index=False)
    with open("verses_dict.json", "w") as file:
        json.dump(max_verses, file, indent=4)
    
    return verse_dict


if __name__ == '__main__':
    assert len(sys.argv) == 5, \
        f'USAGE: python3 {sys.argv[0]} output_filename bible_directory plot_bible_length plots_save_directory'
    output_filename = sys.argv[1]      # File where output from previous experiments is stored
    bible_directory = sys.argv[2]      # Directory where the different Bible translations are stored 
    plot_bible_length = sys.argv[3]    # Plot the length of the different Bible translations
    plots_save_directory = sys.argv[4] # Folder where verse distribution plots will be saved

    verses_dict = create_full_info_file(bible_directory, output_filename)

    if plot_bible_length == "True":
        plot_bible_lengths(verses_dict, plots_save_directory)