from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
import json
import sys


bookdict = {40: 'Matthew', 41:'Mark', 42: "Luke", 43: "John", 44:"Acts", 66:"Revelation"}

def remove_outliers(data: pd.DataFrame, order_string: str, structure_string: str, z_threshold = 3) -> pd.DataFrame:
    """
    Remove outliers using z-scores, return data frame with outliers removes
    """
    data = data.reset_index(drop=True)
    z_order = np.abs(stats.zscore(data[order_string]))
    z_structure = np.abs(stats.zscore(data[structure_string]))
    outlier_indices = np.where((z_order > z_threshold) | (z_structure > z_threshold))[0]
    no_outliers = data.drop(outlier_indices).reset_index(drop=True)

    return no_outliers


def fit_inverse_function(dataframe: pd.DataFrame, plot: bool, plotname: str, order_string: str, structure_string: str, pdf):
    """
    Fit inverse function to the data and plot if wanted
    """

    def inverse_function(x, a, b):
        """
        The inverse function itself
        """
        return a / x + b
    
    def compute_r2(o_data, s_data, a_opt, b_opt):
        """
        Compute r2 value
        """
        y_pred = inverse_function(o_data, a_opt, b_opt)
        ss_res = np.sum((s_data - y_pred) ** 2)  
        ss_tot = np.sum((s_data - np.mean(s_data)) ** 2) 

        return (1 - (ss_res / ss_tot))

    o_data = dataframe[order_string].values
    s_data = dataframe[structure_string].values

    try:
        a0 = (max(s_data) - min(s_data)) * min(o_data)
        b0 = min(s_data)
        params, _ = curve_fit(inverse_function, o_data, s_data, p0=[a0, b0])
    except RuntimeError as e:
        return None

    a_opt, b_opt = params

    spearman_corr, _ = spearmanr(s_data, o_data)
    r2 = compute_r2(o_data, s_data, a_opt, b_opt)

    if plot:
        x_fit = np.linspace(min(o_data), max(o_data), 100)
        y_fit = inverse_function(x_fit, a_opt, b_opt)

        unit = order_string[8:]
        if unit == "":
            unit = "character"

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=o_data, y=s_data, label="Data")
        plt.plot(x_fit, y_fit, color='red', label=f"Fit: $f(x) = {a_opt:.3f}/x + {b_opt:.3f}$")
        plt.xlim(min(o_data), max(o_data))
        plt.ylim(min(s_data), max(s_data))
        plt.xlabel(f"D_order (in bits per {unit})")
        plt.ylabel(f"D_structure (in bits per {unit})")
        plt.title(f"Order vs. structure. Bible: {plotname}.")
        plt.legend()
        plt.figtext(0.5, -0.05, f"n = {len(dataframe)}, spearman r = {spearman_corr}, R^2 = {r2}", ha="center", fontsize=12)
        pdf.savefig(bbox_inches='tight')
        plt.close()

    return a_opt, b_opt


def fit_linear_function(dataframe: pd.DataFrame, plot: bool, plotname: str, order_string: str, structure_string: str, pdf):
    """
    Fit linear function to the data and plot if wanted
    """

    def linear_function(x, a, b):
        """
        The linear function itself
        """
        return a * x + b
    
    def compute_r2(o_data, s_data, a_opt, b_opt):
        """
        Compute r2 value
        """
        y_pred = linear_function(o_data, a_opt, b_opt)
        ss_res = np.sum((s_data - y_pred) ** 2)  
        ss_tot = np.sum((s_data - np.mean(s_data)) ** 2) 

        return (1 - (ss_res / ss_tot))

    o_data = dataframe[order_string].values
    s_data = dataframe[structure_string].values

    try:
        a0 = (max(s_data) - min(s_data)) * min(o_data)
        b0 = min(s_data)
        params, _ = curve_fit(linear_function, o_data, s_data, p0=[a0, b0])

    except RuntimeError as e:
        return None

    a_opt, b_opt = params

    spearman_corr, _ = spearmanr(s_data, o_data)
    r2 = compute_r2(o_data, s_data, a_opt, b_opt)

    if plot:
        x_fit = np.linspace(min(o_data), max(o_data), 100)
        y_fit = linear_function(x_fit, a_opt, b_opt)

        unit = order_string[8:]
        if unit == "":
            unit = "character"

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=o_data, y=s_data, label="Data")
        plt.plot(x_fit, y_fit, color='red', label=f"Fit: $f(x) = {a_opt:.3f}/x + {b_opt:.3f}$")
        plt.xlim(min(o_data), max(o_data))
        plt.ylim(min(s_data), max(s_data))
        plt.xlabel(f"D_order (in bits per {unit})")
        plt.ylabel(f"D_structure (in bits per {unit})")
        plt.title(f"Order vs. structure. {plotname}.")
        plt.legend()
        plt.figtext(0.8, -1, f"n = {len(dataframe)}, spearman r = {spearman_corr}, R^2 = {r2}", wrap=True, horizontalalignment='center', fontsize=10)
        pdf.savefig(bbox_inches='tight')
        plt.close()

    return a_opt, b_opt
  

def plot_histograms(a_values: list, b_values: list, average_a: list, average_b: list, book: int, pdf):
    """
    Plot the values of a an b in histograms
    """
    plt.hist(a_values, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(x=average_a[book])
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of a values for the bible book {bookdict[book]}')
    pdf.savefig()
    plt.close()

    plt.hist(b_values, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(x=average_b[book])
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of b values for the bible book {bookdict[book]}')
    pdf.savefig()
    plt.close()


def create_all_histogram_plots(df: pd.DataFrame, book_a_values: list, book_b_values: list, pdf):
    """
    Calculate the a and b values of fitted inverse functions
    """
    bible_grouped = df.groupby('book_id')

    for book_id, data in bible_grouped:
        data = data[data['verse_length'] >= (max_verses_dict[str(book_id)] * 0.8)]  #uncomment when looking at books with certain amount of verses
        list_of_a = []
        list_of_b = []
        data = remove_outliers(data, "D_order", "D_structure")

        grouped = data.groupby('bible')

        for _, combination in grouped:
            parameters = fit_inverse_function(combination, False, book_id, "D_order", "D_structure", pdf)
            if parameters:
                a, b = parameters
                list_of_a.append(a)
                list_of_b.append(b)
        
        plot_histograms(list_of_a, list_of_b, book_a_values, book_b_values, book_id, pdf)


def koplenig_plots(df: pd.DataFrame, max_verses: dict, order_string: str, structure_string: str, pdf, plot=True):
    """
    Recreate Koplenig et al.'s plots with 0 splits
    """

    book_a_values = {}
    book_b_values = {}

    book_grouped = df.groupby('book_id')

    for book_id, data in book_grouped:
        data = data[data['verse_length'] >= (max_verses[str(book_id)] * 0.8)] # exclude translations with less than 80% of the verses
        no_outliers = remove_outliers(data, order_string, structure_string)
        averages = no_outliers.groupby('bible_language')[[order_string, structure_string]].mean().reset_index()
        a_all, b_all = fit_inverse_function(averages, plot, bookdict[book_id], order_string, structure_string, pdf)

        book_a_values[book_id] = a_all
        book_b_values[book_id] = b_all
    
    return book_a_values, book_b_values


def histogram_values(order_values: list, structure_values: list, book: int, pdf, unit: str):
    """
    Plot the order and structure values with mean and standard deviation
    """
    # Order values
    mean_order = np.mean(order_values)
    std_order = np.std(order_values)
    plt.hist(order_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_order, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_order:.2f}')
    plt.axvline(mean_order + std_order, color='green', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_order:.2f}')
    plt.axvline(mean_order - std_order, color='green', linestyle='dashed', linewidth=1)
    plt.xlabel(f'Order Values (in bits per {unit})')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of order values (in bits per {unit}) for the bible book {bookdict[book]}')
    plt.legend()
    pdf.savefig()
    plt.close()
    
    # Structure values
    mean_structure = np.mean(structure_values)
    std_structure = np.std(structure_values)
    plt.hist(structure_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_structure, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_structure:.2f}')
    plt.axvline(mean_structure + std_structure, color='green', linestyle='dashed', linewidth=1, label=f'Std Dev: {std_structure:.2f}')
    plt.axvline(mean_structure - std_structure, color='green', linestyle='dashed', linewidth=1)
    plt.xlabel(f'Structure Values (in bits per {unit})')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of structure values (in bits per {unit}) for the bible book {bookdict[book]}')
    plt.legend()
    pdf.savefig()
    plt.close()


def plot_values_all_books(df: pd.DataFrame, max_verses_dict: dict, order_value: str, structure_value: str, pdf):
    """
    Plot the histograms of order and structure values for all books.
    """
    bible_grouped = df.groupby('book_id')
    unit = order_value[8:]
    if unit == "":
        unit = "character"

    for book, data in bible_grouped:
        data = data[data['verse_length'] >= (max_verses_dict[str(book)] * 0.8)]
        no_outliers = remove_outliers(data, order_value, structure_value)
        histogram_values(no_outliers[order_value], no_outliers[structure_value], book, pdf, unit)


if __name__ == '__main__':
    assert len(sys.argv) == 4, \
        f'USAGE: python3 {sys.argv[0]} output_file maximum_verses save_filepath'
    output_file = sys.argv[1]        # Output csv file
    maximum_verses = sys.argv[2]     # JSON file with dictionary of max amount of verses
    save_filepath = sys.argv[3]      # Folder where plots will be saved

    df = pd.read_csv(output_file)
    zero_df = df[df['iter_id'] == 0]   # take data with no splitting/pasting

    with open(maximum_verses, "r") as file:
        max_verses_dict = json.load(file)

    output_pdf_path = os.path.join(save_filepath, f"Entropy_plots.pdf")
    with PdfPages(output_pdf_path) as pdf:
        for unit in ["_word", "_verse", ""]:
            a, b = koplenig_plots(zero_df, max_verses_dict, f"D_order{unit}", f"D_structure{unit}", pdf)
            plot_values_all_books(zero_df, max_verses_dict, f"D_order{unit}", f"D_structure{unit}", pdf)

        create_all_histogram_plots(df, a, b, pdf)