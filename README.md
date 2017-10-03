Overview
============
A convolutional neural network written in python using tensorflow, that is designed to solve sudoku puzzles.
Note: Sudoku solution needs a renforced learning approach, so that the existing numbers are not modified.

Training Datsource :
============
[1 million](https://www.kaggle.com/bryanpark/sudoku) Sudoku games by [Bryan Park](https://www.kaggle.com/bryanpark)
A tarball for the above data is included and should work just fine.
Otherwise download the above data source and place the csv file in the `data/` subdirectory of current repo. Downloading the data source requies sign in with a kaggle account.

Dependencies :
============
* [numpy](www.numpy.org)
* [scikit-learn](scikit-learn.org)
* [tensorflow](www.tensorflow.org)

Usage :
============
from a docker container or local installation run, `python sudoku_nn.py`
