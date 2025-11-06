# Noun-Noun Compounds

This repository contains multiple attempts to study entropy using the bible, including training LSTMs, computing perplexities with GPT-2, and performing word-pasting and word-splitting experiments. A much more limited of this repository, including only the word-pasting experiments, is in the BibleWordPasting repository.

2025 Update: this paper has been published in Coling-Rel and, therefore, there is no need to maintain it anonymous anymore. However, the link provided in the camera-ready paper is anonymous. The published paper can be found [here](https://aclanthology.org/2025.clrel-1.9/). The full reference is:

Mosteiro, P., & Blasi, D. (2025). Word boundaries and the morphology-syntax trade-off. In S. Yagi, S. Yagi, M. Sawalha, B. A. Shawar, A. T. AlShdaifat, N. Abbas, & Organizers (Eds.), *Proceedings of the New Horizons in Computational Linguistics for Religious Texts* (pp. 86–93). Association for Computational Linguistics. [https://aclanthology.org/2025.clrel-1.9/](https://aclanthology.org/2025.clrel-1.9/)

The GitHub repository is WordOrderBibles, owned by PabloMosUU.
This repository also supports the paper *West Germanic noun-noun compounds and the morphology-syntax trade-off*, by Mosteiro, Blasi, and Paperno. The main code used for this analysis is in `nn_pasting.py` and `NounNounCompounds/11_final_paper_plots.py`. 

## Requirements

Python packages required to run this code can be found in `requirements.txt`. You may install them all at once with:

	pip install -r requirements.txt

### Mismatcher

The mismatcher [1] is a Java program used in previous work to compute, for each position in a text, the shortest unseen substring at that position [2]. Those values are then used to compute an approximation to the entropy [3].

## Running this code

The main entry points are `word_pasting.py` and `word_splitting.py`. The usage of these programs is:

	python word_pasting.py bible_filename temp_dir output_filename mismatcher_filename
	
	python word_splitting.py bible_filename temp_dir output_filename mismatcher_filename n_merges_full

The parameters are:

* `bible_filename`: the full path of the file containing a single translation of the bible coming from the Parallel Bible Corpus (or any compatible format)
* `temp_dir`: a directory in which temporary files (not the output) will be saved
* `output_filename`: the directory in which output files will be saved
* `mismatcher_filename`: the full path of the Java binary used to compute the lengths of the shortest unseen strings at each position in the bible translation file (see details above)
* `n_merges_full`: the number of maximum merges that should be input to the BPE algorithm (see details below)

### `n_merges_full`

`n_merges_full` is the number of merges to train the BPE tokenizer aiming to build the entire merge history. In [4] we used 10000 for most bibles. The program will print a warning if this number is not high enough to generate the entire merge history. For those bibles, we recommend using `n_merges_full` = 30000.

## References

[1] Koplenig, A., Meyer, P., Wolfer, S., & Müller-Spitzer, C. (2017). Replication Data for: The statistical trade-off between word order and word structure – large-scale evidence for the principle of least effort. https://doi.org/10.7910/DVN/8KH0GB

[2] Koplenig, A., Meyer, P., Wolfer, S., & Müller-Spitzer, C. (2017). The statistical trade-off between word order and word structure – Large-scale evidence for the principle of least effort. PLOS ONE, 12(3), 1–25. https://doi.org/10.1371/journal.pone.0173614

[3] Kontoyiannis, I., Algoet, P., Suhov, Y., & Wyner, A. (1998). Nonparametric Entropy Estimation for Stationary Processesand Random Fields, with Applications to English Text. Information Theory, IEEE Transactions On, 44, 1319–1327. https://doi.org/10.1109/18.669425

[4] Mosteiro, P., & Blasi, D. (2025). Word boundaries and the morphology-syntax trade-off. In S. Yagi, S. Yagi, M. Sawalha, B. A. Shawar, A. T. AlShdaifat, N. Abbas, & Organizers (Eds.), Proceedings of the New Horizons in Computational Linguistics for Religious Texts (pp. 86–93). Association for Computational Linguistics. https://aclanthology.org/2025.clrel-1.9/

