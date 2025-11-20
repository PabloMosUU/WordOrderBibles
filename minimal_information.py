import ast
import sys, os
import numpy as np
import pandas as pd 
import koplenig_plots as kp
import create_full_information_csv as cf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

bookdict = {'40': 'Matthew', '41':'Mark', '42': "Luke", '43': "John", '44':"Acts", '66':"Revelation", "full_bible": "Full Bible"}

def find_min_information_point(dataframe: pd.DataFrame):
    """
    Find the data point in the dataframe where the sum D_order + D_structure is minimal.
    """
    sums = dataframe["D_order"] + dataframe["D_structure"]
    min_index = sums.idxmin()
    min_row = dataframe.loc[min_index]

    return min_row["D_order"], min_row["D_structure"], min_row


def find_closest_slope_point(dataframe: pd.DataFrame, a_opt: float, b_opt: float):
    """
    Find the data point (x, y) in the dataframe closest to where the slope of y = a/x + b is -1.
    This happens at x = sqrt(a), y = sqrt(a) + b.
    """
    x_target = np.sqrt(a_opt)
    y_target = x_target + b_opt

    x_data = dataframe["D_order"].values
    y_data = dataframe["D_structure"].values

    distances = np.sqrt((x_data - x_target)**2 + (y_data - y_target)**2)
    closest_index = np.argmin(distances)

    closest_x = x_data[closest_index]
    closest_y = y_data[closest_index]
    return closest_x, closest_y, dataframe.iloc[closest_index]


def plotting_optimal_points(dataframe: pd.DataFrame, pdf):
    """
    Plot the data points with optimal information for all books
    """
    for book in dataframe['book'].unique():
        book_df = dataframe[dataframe['book'] == book]
        points = []

        for translation in book_df['bible'].unique():
            translation_df = book_df[book_df['bible'] == translation]

            result = kp.fit_inverse_function(translation_df, False, book, "D_order", "D_structure", pdf)
            if result is None:
                print(f"Skipping {translation} in {book} (fit failed)")
                continue

            a, b = result
            #ord, str, _ = find_min_information_point(translation_df)
            ord, str, _ = find_closest_slope_point(translation_df, a, b)

            points.append((ord, str))
            df_points = pd.DataFrame(points, columns=['ord', 'str'])

        kp.fit_inverse_function(df_points, True, book, 'ord', 'str', pdf)


def plotting_optimal_points_some_languages(dataframe: pd.DataFrame):
    """
    Plot the optimal information points for the languages discussed in Koplenig et al.
    """
    dataframe = dataframe[dataframe["bible_language"].isin(["chr", "cmn", "deu", "eng", "esk", "grc", "mya", "vie", "xuo", "zul", "tam", "qvw"])]

    for book in dataframe['book_id'].unique():
        book_df = dataframe[dataframe['book_id'] == book]

        plt.figure(figsize=(8, 6))
        plt.title(f"Avg closest points to 45° slope for each language in '{bookdict[book]}'")
        plt.xlabel("D_order (bpc)")
        plt.ylabel("D_structure (bpc)")

        for language in book_df['bible_language'].unique():
            lang_df = book_df[book_df['bible_language'] == language]
            points = []

            for translation in lang_df['bible'].unique():
                translation_df = lang_df[lang_df['bible'] == translation]

                result = kp.fit_inverse_function(translation_df, False, "Splits", "D_order", "D_structure")
                if result is None:
                    continue

                a,b  = result
                ord_val, str_val, _ = find_min_information_point(translation_df)
                points.append((ord_val, str_val))

            if points:
                avg_ord = sum(p[0] for p in points) / len(points)
                avg_str = sum(p[1] for p in points) / len(points)
                plt.scatter(avg_ord, avg_str, label=language)

        plt.legend(title="Language", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


def one_translation_merging(df: pd.DataFrame, pdf):
    """
    Provides optimal information points and plots for one translation
    """
    bible_grouped = df.groupby('book_id')

    for book, bible_df in bible_grouped:
        a,b = kp.fit_inverse_function(bible_df, True, bookdict[str(book)], 'D_order', 'D_structure', pdf)
        #kp.fit_linear_function(df_output, True, "Full Bible", 'D_order', 'D_structure', pdf) #Uncomment if linear function is wanted
        
        # Print optimal points
        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        ax.axis('off')
        _, _, slope_point = find_closest_slope_point(bible_df, a , b)
        _, _, min_point = find_min_information_point(bible_df)

        text = (
            f"Book: {bookdict[str(book)]}\n\n"
            f"Optimal point based on 45 degree angle:\n{slope_point}\n\n"
            f"Minimal information point (D_order + D_structure):\n{min_point}"
        )

        ax.text(0.01, 0.95, text, fontsize=9, va='top')
        pdf.savefig(fig)
        plt.close(fig)

def create_output_df(filename: str):
    """
    Create output dataframe from entropy dictionary in file.
    """
    with open(filename, 'r') as file:
        data_str = file.read()
    data_dict = ast.literal_eval(data_str)

    all_dfs = []
    final_df = pd.DataFrame

    for book, bible_data in data_dict.items(): 
        orig_values = []
        shuffled_values = []
        masked_values = []
        keys = []

        for key in sorted(bible_data.keys(), key=lambda x: int(x)):
            entry = bible_data[key]
            keys.append(key)
            orig_values.append(entry['orig'])
            shuffled_values.append(entry['shuffled'])
            masked_values.append(entry['masked'])

        D_order = [s - o for o, s in zip(orig_values, shuffled_values)]
        D_structure = [m - o for o, m in zip(orig_values, masked_values)]

        df_output = pd.DataFrame({
            'book_id' : book,
            'iterations' : keys,
            'D_order': D_order,
            'D_structure': D_structure})
        
        all_dfs.append(df_output)
        
    final_df = pd.concat(all_dfs, ignore_index=True)

    return final_df


def overlay_plots(pos_df: pd.DataFrame, normal_df: pd.DataFrame, plotname,  pdf):
    """
    Create plots with both the POS-based merging and the normal merging.
    """
    group_normal_df = normal_df.groupby('book_id')

    for book_id, normal_df in group_normal_df:
        pos_book_df = pos_df[pos_df['book_id'] == str(book_id)]
        plt.scatter(normal_df['D_order'].values, normal_df['D_structure'].values, label='Normal', s=5, color='blue')
        plt.scatter(pos_book_df['D_order'].values, pos_book_df['D_structure'].values, label='POS', s= 5, color='red')
        plt.xlabel('D_order')
        plt.ylabel('D_structure')
        plt.title(f'Overlay of normal and pos merges for {bookdict[str(book_id)]} {plotname}')
        plt.legend()
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    assert len(sys.argv) == 5, \
        f'USAGE: python3 {sys.argv[0]} output_file output_file_pos save_filepath translation_name'
    output_file = sys.argv[1]        # Output csv file
    output_file_pos = sys.argv[2]    # Output name of pos-based merging
    save_filepath = sys.argv[3]      # Folder where plots will be saved
    translation_name = sys.argv[4]   # Translation to print more details on normal vs. pos-based merging

    df = cf.read_csv_file(output_file)
    pos_df = create_output_df(output_file_pos)
    translation_df = df[df['bible'] == translation_name]

    output_pdf_path = os.path.join(save_filepath, f"minimal_information_plots_fullbible_deu.pdf")
    with PdfPages(output_pdf_path) as pdf:
        plotting_optimal_points(df, pdf)
        #plotting_optimal_points_some_languages(df) # Uncomment if plots for a limited number of languages is wanted
        one_translation_merging(df, pdf)
        one_translation_merging(pos_df, pdf)
        overlay_plots(pos_df, df, translation_name, pdf)
