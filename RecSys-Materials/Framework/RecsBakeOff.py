# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from MovieLens import MovieLens
from surprise import SVD
from surprise import NormalPredictor
from Evaluator import Evaluator

import random
import numpy as np
# RecsBakeOff file cuối cùng, chạy tất cả các thứ cần thiết
# Đầu tiên là load data, sau đó tạo 1 Evaluator và thêm các Algo vào, cuối cùng Evalutate

# RecommenderMetrics (Tính toán các thứ như MAE, RMSE, Hitrate, Density, Novelty)
# + EvaluationData (Class để chia data thành train set, test set)
# > EvaluatedAlgorithm

# EvaluatedAlgorithm + Các hàm khác trong Algobase > Evaluator
# Evaluator (thêm các Algo vào + Evaluate) + MovieLens (load data) > RecsBakeOff

def LoadMovieLensData(): #return movies data dạng dataset cx như Ranking
    # sử dụng class MovieLems.py các hàm trong MovieLems.py
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (data, rankings)

np.random.seed(0) # đảm bảo rằng mỗi lần chạy thì đều ra cùng 1 kết quả
random.seed(0)

# Load up common data set for the recommender algorithms
# Gán 2 giá trị return cho hàm def bên trên
(evaluationData, rankings) = LoadMovieLensData()

# Construct an Evaluator to, you know, evaluate them
# Tạo 1 instance cho class Evaluator, cần 2 tham số đầu vào cho __init__
evaluator = Evaluator(evaluationData, rankings)

# Throw in an SVD recommender
SVDAlgorithm = SVD(random_state=10)
evaluator.AddAlgorithm(SVDAlgorithm, "SVD")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")


# Fight!
evaluator.Evaluate(True)

