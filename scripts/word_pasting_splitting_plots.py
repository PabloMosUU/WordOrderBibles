from wordorderbibles import util
import sys
import pandas as pd

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError(f'Usage: python {sys.argv[0]} <full_results_file> <output_dir>')
    # full_results_file is a CSV file containing D_order and D_structure for 1 or more bibles, for 1 or more books,
    # for both pasting and splitting
    full_results_file = sys.argv[1]
    output_dir = sys.argv[2]
    df = pd.read_csv(full_results_file)
    for bible, grp in df.groupby('bible'):
        util.make_plot_(grp, str(bible), output_dir)
