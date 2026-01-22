import sys
import os

from util import to_csv

if __name__ == '__main__':
    assert len(sys.argv) == 2, f'USAGE: python3 {sys.argv[0]} json_file_dir'
    filedir = os.path.join(os.getcwd(), sys.argv[1])
    print('filedir:', filedir)
    files = os.listdir(filedir)
    print(files)
    json_files = [el for el in files if el.endswith('json')]
    print(json_files)
    csv_files = set([el for el in files if el.endswith('csv')])
    print(csv_files)
    for a_json_file in json_files:
        if a_json_file.replace('json', 'csv') in csv_files:
            print('skip', a_json_file)
        else:
            print('convert', a_json_file)
            to_csv(os.path.join(filedir, a_json_file))
