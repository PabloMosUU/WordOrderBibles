import matplotlib.pyplot as plt
import csv
import sys

def plot(data: list) -> None:
    unique_books = set([row['book'] for row in data])
    for book_name in unique_books:
        book_data = [row for row in data if row['book'] == book_name]
        x = [float(row['D_order']) for row in book_data]
        y = [float(row['D_structure']) for row in book_data]
        labels = [row['iter_id'] for row in book_data]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        plt.xlabel('Word order information')
        plt.ylabel('Word structure information')
        plt.title(f'{book_name}')
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]), rotation=45)
    plt.show()

def read(filename: str) -> list:
    with open(filename, newline='') as csvfile:
        entropy_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        rows = list(entropy_reader)
    return rows

def parse(basic_rows: list) -> list:
    header = basic_rows[0]
    body = basic_rows[1:]
    return [{header[i]: el for i, el in enumerate(row)} for row in body]

if __name__ == '__main__':
    assert len(sys.argv) == 2, f'USAGE: python3 {sys.argv[0]} csv_file_name'
    csv_filename = sys.argv[1]
    csv_rows = read(csv_filename)
    structured_data = parse(csv_rows)
    plot(structured_data)
