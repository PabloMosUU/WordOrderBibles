import numpy as np
import matplotlib.pyplot as plt
import sys, os
from matplotlib.backends.backend_pdf import PdfPages

bookid = {40:"Matthew", 41:"Mark", 42:"Luke", 43:"John", 44:"Acts", 66:"Revelation", "fullbible" : "Full Bible"}

def convert_mismatcher_file(input_file: str):
    """
    Reads mismatcher file and returns list of longest unseen sequence at every index.
    """
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
   
    li_values = []

    for line in lines:
        li_values.append(line.strip().split('\t')[-1])

    return li_values


def calcualate_entropy(lines: list) -> list:
    """
    Calculates entropy using the entropy formula in Koplenig et al.
    """
    my_sum = 0
    entropies = []

    for i, line in enumerate (lines[1:]):
        my_sum += int(line) / np.log2(i + 2) # i + 2 is index we are currently at (counted from first character is index 1)
        entropies.append((i + 2) / my_sum) # i + 2 is length of string seen

    return entropies


def plot_entropies(entropy_list: list, title: str, pdf: PdfPages):
    """
    Plots the entropies at the different indices and saves them to a PDF.
    """
    indices = list(range(len(entropy_list)))

    plt.figure()
    plt.plot(indices, entropy_list)
    plt.xlabel('Index')
    plt.ylabel('Entropy Value (in bpc)')
    plt.title(fr'Plot of Entropy value at given indices for {title}')
    
    pdf.savefig()  
    plt.close()


def run_convergence_on_translation(folder: str, translation: str, save_filepath: str, plot_str_ord: bool):
    """
    Plot entropy convergence per book and save all plots into one PDF.
    """
    output_pdf_path = os.path.join(save_filepath, f"{translation}_entropies.pdf")
    with PdfPages(output_pdf_path) as pdf:
        for i in [40, 41, 42, 43, 44, 66, "fullbible"]:
            file = fr"{folder}\{translation[:-4]}_orig.txt_{i}_v0_mismatcher"
            l_i_lines = convert_mismatcher_file(file)
            entropies = calcualate_entropy(l_i_lines)
            plot_entropies(entropies, bookid[i], pdf)

            if plot_str_ord == 'True':
                plot_str_ord_convergence(folder, translation, i, pdf)


def plot_str_ord_convergence(folder: str, translation: str, book: str, pdf):
    """
    Plot the convergence of D_order and D_structure for a book in a translation.
    """
    file_orig = fr"{folder}\{translation[:-4]}_orig.txt_{book}_v0_mismatcher"
    file_shuffled = fr"{folder}\{translation[:-4]}_shuffled.txt_{book}_v0_mismatcher"
    file_masked = fr"{folder}\{translation[:-4]}_masked.txt_{book}_v0_mismatcher"

    entropy_orig = calcualate_entropy(convert_mismatcher_file(file_orig))
    entropy_shuffled = calcualate_entropy(convert_mismatcher_file(file_shuffled))
    entropy_masked = calcualate_entropy(convert_mismatcher_file(file_masked))

    D_order = [e_shuffled - e_orig for e_shuffled, e_orig in zip(entropy_shuffled, entropy_orig)]
    D_structure = [e_masked - e_orig for e_masked, e_orig in zip(entropy_masked, entropy_orig)]

    plot_entropies(D_order, fr"D_order {bookid[book]}", pdf)
    plot_entropies(D_structure, fr"D_structure {bookid[book]}", pdf)


if __name__ == '__main__':
    assert len(sys.argv) == 5, \
        f'USAGE: python3 {sys.argv[0]} mismatcher_output_folder translation_file output_file plot_order+structure'
    mismatcher_output = sys.argv[1]  # Folder where mismatcher output is saved
    bible_translation = sys.argv[2]  # Name of the Bible translation text file
    save_filepath = sys.argv[3]      # File name where output is stored
    plot_ord_str = sys.argv[4]       # Plot D_order and D_structure convergence

    run_convergence_on_translation(mismatcher_output, bible_translation, save_filepath, plot_ord_str)