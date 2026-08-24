import sys
from wordorderbibles import util

if __name__ == '__main__':
    assert len(sys.argv) == 2, f'USAGE: python3 {sys.argv[0]} json_file_name'
    filename = sys.argv[1]
    util.to_csv(filename)
