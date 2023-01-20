
from MovieLens import MovieLens
from AutoRecAlgorithm import AutoRecAlgorithm
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

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#Autoencoder
AutoRec = AutoRecAlgorithm()
evaluator.AddAlgorithm(AutoRec, "AutoRec")

Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
