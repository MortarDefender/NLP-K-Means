import sys
import json
import pandas as pd
from time import time
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer as Downscale

sys.path.append('../')

from src.kmeans import Kmeans
from src.kmeansPlus import KmeansPlus


def getLabels(fileName):
    dataFile = pd.read_csv(fileName, delimiter="\t")
    dataFile = dataFile.set_axis(["labels", "data"], axis = 1, inplace = False)
    allLabels = dataFile["labels"]

    return allLabels, len(set(allLabels))


def getFeatureVectors(fileName):
    dataFile = pd.read_csv(fileName, delimiter="\t")
    dataFile = dataFile.set_axis(["labels", "data"], axis = 1, inplace = False)
    data = dataFile["data"]
    vectorizer = Downscale(max_features = 20000)
    tf_idf = vectorizer.fit_transform(data)
    # tf_idf_norm = normalize(tf_idf)
    # tf_idf_array = tf_idf_norm.toarray()
    tf_idf_array = tf_idf.toarray()

    return tf_idf_array


def kmeans_cluster_and_evaluate(data_file, plus = False, debug = False):
    print('starting kmeans clustering and evaluation with', data_file)
    
    t = time()
    roundsAmount = 200
    evaluation_results = {'mean_RI_score': 0.0, 'mean_ARI_score': 0.0}
    
    data = getFeatureVectors(data_file)
    labels, labelsAmount = getLabels(data_file)
    
    if plus:
        classfier = KmeansPlus(labelsAmount)
    else:
        classfier = Kmeans(labelsAmount)
    
    avregeRi, avregeAri = 0, 0

    for i in range(roundsAmount):
        classfier.fit(data)
        
        ri, ari = classfier.accuracy(labels)
        
        avregeRi += ri
        avregeAri += ari
        
        if debug and i % 50 == 0:
            print(f"{i}. {ri} - {ari}  ==> {time() - t} s")

    classfier.plot()

    evaluation_results['mean_RI_score'] = avregeRi / roundsAmount * 100
    evaluation_results['mean_ARI_score'] = avregeAri / roundsAmount * 100

    return evaluation_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = kmeans_cluster_and_evaluate(config['data'])
    # results = kmeans_cluster_and_evaluate(config['small-data'])

    for k, v in results.items():
        print(k, v)
