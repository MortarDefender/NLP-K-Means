[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

# NLP-K-Means
implementation of k-means and k-means++ algorithms

## System Requirements:
- python 3.6 or heighr
- sklearn library
- pandas library
- numpy library
- matplotlib library

## Installation:
installation can be done using conda.

```cmd
conda activate
python setup.py install
```

## Run:
```python
from kmeansPlus import KmeansPlus

classfier = KmeansPlus(labelsAmount)
classfier.fit(data)
ri_score, ari_score = classfier.accuracy(labels)
classfier.plot()
```

or

```python
from kmeans import Kmeans

classfier = Kmeans(labelsAmount)
classfier.fit(data)
ri_score, ari_score = classfier.accuracy(labels)
classfier.plot()
```

# The Task At Hand:

Clustering algorithms work with text representations – feature vectors. Use the same approach for
feature extraction you implemented in assignment #2: each document (sentence in this case) is
represented by its tf-idf vector. The quality of your feature vectors is likely to affect the results.

After performing clustering, evaluate you results against the provided labels: the RandIndex (RI) metric,
and its adjusted for chance version – Adjusted RandIndex (ARI). Please read through the ARI metric
documentation on the web and make sure you understand how it improves the basic RI.

    ARI is considered a more reliable metric since RI is biased towards large number of clusters. When there
    are a many clusters, the chances that a pair of items in both your result and the ground truth are in
    different clusters are high, and this is still counted as a concordant event in RI.

# Kmeans Test Examples:

<img width="45%" height="250px" src="/Demo/kmeansLargeData.png" /> <img width="45%" height="250px" src="/Demo/kmeansSmallData.png" /> 


# Kmeans Plus Test Examples:

<img width="45%" height="250px" src="/Demo/kmeansPlusLargeData.png" /> <img width="45%" height="250px" src="/Demo/kmeansPlusSmallData.png" />
