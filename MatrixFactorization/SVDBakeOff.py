from MovieLens import MovieLens
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

(ml, evaluationData, rankings) = LoadMovieLensData()

evaluator = Evaluator(evaluationData, rankings)

SVD = SVD()
evaluator.AddAlgorithm(SVD, "SVD")

SVDPlusPlus = SVDpp()
evaluator.AddAlgorithm(SVDPlusPlus, "SVD++")

Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
