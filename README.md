# Machine Learning?!

This repository contains some of my efforts to learn how machine learning works on a more technical level.

## `embedding-tester.py`

This script programmatically clusters requirements based on their semantic similarity. This might be useful for drafting application architecture given a list of plain-text requirements.

The idea is to demonstrate two machine-learning techniques:
- Sentence-level embedding
- Semantic clustering

## Setup

- Start with Python 3.
- Install the necessary Python libraries as follows:

```sh
python -m venv .
. bin/activate
pip install sentence-transformers faiss-cpu scikit-learn numpy
```

- Alternatively, run the setup script (does exactly what you see in the shell script above)
- `setup-venv-and-dependencies.sh`

## Usage

The list of requirements lives in the file `early-bird-requirements.txt`. These requirements come from the "EarlyBird Case Study" as provided to students at the FH Technikum Wien for the Software Architecture module.
The script reads this file, generates sentence-level embeddings, puts them into a vector database, then clusters the sentences. The `n_clusters` variable can be changed as needed.

The script downloads model files the first time they're used, meaning the first run of the script will take much longer than subsequent runs.

Example usage:
``` sh
$ cd /path/to/this/repository/
$ . /bin/activate
(embeddings-and-clustering) $ python embedding-tester.py
```
