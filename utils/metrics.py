# This file contains basic metrics used for evaluation
# Author: Stefan Kahl, 2018, Chemnitz University of Technology

from sklearn.metrics import average_precision_score
from sklearn.metrics import label_ranking_average_precision_score

def averagePrecision(prediction, target):

    # Calculate average precision for every sample
    return average_precision_score(target, prediction, average='samples')

def lrap(prediction, target):

    # Calculate the label ranking average precision (LRAP) for every sample
    return label_ranking_average_precision_score(target, prediction)

