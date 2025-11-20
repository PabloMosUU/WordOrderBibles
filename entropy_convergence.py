import sys
import word_pasting as wp

if __name__ == '__main__':
    assert len(sys.argv) == 4, \
        f'USAGE: python {sys.argv[0]} bible_filename temp_dir mismatcher_filename'
    bible_filename = sys.argv[1]  # The bible filename
    temp_dir = sys.argv[2]  # The directory where Mismatcher files are saved
    mismatcher_file = sys.argv[3]  # The filename of the mismatcher jar

    merge_steps = {0}

    wp.run_word_pasting_all(bible_filename,
                            lowercase=True,
                            remove_mismatcher_files=False,
                            truncate_books=False,
                            merge_steps_to_save=merge_steps,
                            output_file_path=temp_dir,
                            mismatcher_path=mismatcher_file)
