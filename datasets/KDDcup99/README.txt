KDD Cup 99 task (logistic regression)
-------------------------------------

A. Obtain data set:

1. Go to https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup99.html
2. Download kddcup.data_10_percent.gz and decompress.
3. This gives the file "kddcup.data_10_percent". Move it into this folder.
4. Run process_kdd.py. This produces the file kddcup_dataset.txt.
5. To get a smaller dataset, take a random subsample (without replacement), e.g. in bash:

      $ cat kddcup_dataset.txt | shuf | head -n 50000 > kddcup_smaller_dataset.txt


--------------------------------

B. Run logistic regression code on this dataset file.

