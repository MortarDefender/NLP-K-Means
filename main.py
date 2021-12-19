import csv
import json

from kmeans import Kmeans
from kmeansPlus import KmeansPlus

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as Downscale


def getLabels(fileName):
    allLabels = []
    
    with open(fileName, "r") as f:
        tsv = csv.reader(f, delimiter="\t")
        
        for line in tsv:
            allLabels.append(line[0])
    
    return allLabels, len(set(allLabels))


def getFeatureVectors(fileName):
    data = []
    
    with open(fileName, "r") as f:
        tsv = csv.reader(f, delimiter="\t")
        
        for line in tsv:
            data.append(line[1])
    
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    arrayOfAppearances = vectorizer.fit_transform(data)
    return Downscale().fit_transform(arrayOfAppearances)


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
    evaluation_results = {'mean_RI_score':  0.0,
                          'mean_ARI_score': 0.0}

    return evaluation_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = kmeans_cluster_and_evaluate(config['data'])

    for k, v in results.items():
        print(k, v)
