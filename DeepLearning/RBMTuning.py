from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV

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

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

print("Searching for best parameters...")
param_grid = {'hiddenDim': [20, 10], 'learningRate': [0.1, 0.01]}
gs = GridSearchCV(RBMAlgorithm, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(evaluationData)

# best RMSE score
print("Best RMSE score attained: ", gs.best_score['rmse'])

print(gs.best_params['rmse'])

evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']
RBMtuned = RBMAlgorithm(hiddenDim = params['hiddenDim'], learningRate = params['learningRate'])
evaluator.AddAlgorithm(RBMtuned, "RBM - Tuned")

RBMUntuned = RBMAlgorithm()
evaluator.AddAlgorithm(RBMUntuned, "RBM - Untuned")

Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
