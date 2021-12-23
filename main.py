import json

from kmeans import Kmeans
from kmeansPlus import KmeansPlus

import pandas as pd
from sklearn.preprocessing import normalize
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer as Downscale
from sklearn.feature_extraction.text import TfidfVectorizer as Downscale


def getLabels(fileName):
    dataFile = pd.read_csv(fileName, delimiter="\t")
    dataFile = dataFile.set_axis(["labels", "data"], axis=1, inplace=False)
    allLabels = dataFile["labels"]

    return allLabels, len(set(allLabels))


def getFeatureVectors(fileName):
    dataFile = pd.read_csv(fileName, delimiter="\t")
    dataFile = dataFile.set_axis(["labels", "data"], axis=1, inplace=False)
    data = dataFile["data"]
    vectorizer = Downscale(stop_words='english',  # tokenizer = tokenize_and_stem,
                           max_features=20000)
    tf_idf = vectorizer.fit_transform(data)
    tf_idf_norm = normalize(tf_idf)
    tf_idf_array = tf_idf_norm.toarray()

    # pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()
    print(tf_idf_array)
    return tf_idf_array


def kmeans_cluster_and_evaluate(data_file):
    # todo: implement this function
    print('starting kmeans clustering and evaluation with', data_file)

    # todo: perform feature extraction from sentences and
    #  write your own kmeans implementation with random centroids initialization
    #  at the next step, enhance the procedure to work with kmeans++ initialization
    #  please use kmeans++ in the final version you submit

    # todo: evaluate against known ground-truth with RI and ARI:
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.rand_score.html and
    #  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

    # todo: fill in the dictionary below with evaluation scores averaged over X runs
    evaluation_results = {'mean_RI_score': 0.0,
                          'mean_ARI_score': 0.0}

    roundsAmount = 1
    data = getFeatureVectors(data_file)
    labels, labelsAmount = getLabels(data_file)

    classfier = KmeansPlus(labelsAmount, data)
    avregeRi, avregeAri = 0, 0

    for i in range(roundsAmount):
        classfier.fit()
       # classfier.plot()
   #     ri, ari = classfier.accuracy(labels)
    #    avregeRi += ri
     #   avregeAri += ari

    evaluation_results['mean_RI_score'] = avregeRi / roundsAmount
    evaluation_results['mean_ARI_score'] = avregeAri / roundsAmount

    return evaluation_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = kmeans_cluster_and_evaluate(config['data'])

    for k, v in results.items():
        print(k, v)