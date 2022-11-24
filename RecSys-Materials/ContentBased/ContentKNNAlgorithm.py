# -*- coding: utf-8 -*-
"""
Created on Fri May  4 13:08:25 2018

@author: Frank
"""

from surprise import AlgoBase
from surprise import PredictionImpossible
from MovieLens import MovieLens
import math
import numpy as np
import heapq

class ContentKNNAlgorithm(AlgoBase):

    def __init__(self, k=40, sim_options={}):
        AlgoBase.__init__(self)
        self.k = k

    def fit(self, trainset):
        # Train an algorithm on a given training set.
        # This method is called by every derived class as the first basic step for training an algorithm.
        # It basically just initializes some internal structures and set the self.trainset attrib
        AlgoBase.fit(self, trainset)

        # Compute item similarity matrix based on content attributes

        # Load up genre, year, miseenscene vectors (array) for every movie
        ml = MovieLens()
        genres = ml.getGenres()
        years = ml.getYears()
        mes = ml.getMiseEnScene()
        
        print("Computing content-based similarity matrix...")
            
        # Compute genre distance for every movie combination as a 2x2 matrix
        # np.zeros(n,m): Tạo 1 ma trận nxm với toan số 0
        # n_items: Total number of items :math:`|I|`.
        # TH này là tat cả các bộ phim đc so sánh từng cặp vs nhau
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
        
        for thisRating in range(self.trainset.n_items):
            # this rating ~ stt của movie (movieID)
            if (thisRating % 100 == 0):
                print(thisRating, " of ", self.trainset.n_items)
                # Hien thị giai đoạn để theo dõi,  Vd 0/200, 100/200

            # so sánh 1 movie vs tất cả những movie còn lai đổ lên (k ss ngược)
            for otherRating in range(thisRating+1, self.trainset.n_items):
                thisMovieID = int(self.trainset.to_raw_iid(thisRating)) # Item's id to Inner' id
                otherMovieID = int(self.trainset.to_raw_iid(otherRating))
                #genreSimilarity = self.computeGenreSimilarity(thisMovieID, otherMovieID, genres)
                yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                #mesSimilarity = self.computeMiseEnSceneSimilarity(thisMovieID, otherMovieID, mes)

                # Access phần tử trong ma trân (movie x movie) self.similarities với [x,y] (<class 'numpy.ndarray'>)
                #self.similarities[thisRating, otherRating] = genreSimilarity * yearSimilarity
                #self.similarities[thisRating, otherRating] = genreSimilarity #chỉ lấy gerne
                self.similarities[thisRating, otherRating] = yearSimilarity # chỉ lấy year

                # Similarity giữa A và B giống B và A
                self.similarities[otherRating, thisRating] = self.similarities[thisRating, otherRating]
        # [[0.         0.         0.... 0.5        0.         0.5]
        #  [0.         0.         0.... 0.         0.         0.]
        # [0.]]

        return self
    
    # Tính độ tg dong của gerne giữa 2 movie dựa vào multi dimension cosine
    def computeGenreSimilarity(self, movie1, movie2, genres):
        genres1 = genres[movie1]
        genres2 = genres[movie2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(genres1)):
            x = genres1[i]
            y = genres2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        
        return sumxy/math.sqrt(sumxx*sumyy)

    # Tính độ tg dong của year giữa 2 movie dựa vào multi dimension cosine
    def computeYearSimilarity(self, movie1, movie2, years):
        diff = abs(years[movie1] - years[movie2])
        sim = math.exp(-diff / 10.0)
        return sim
    
    def computeMiseEnSceneSimilarity(self, movie1, movie2, mes):
        mes1 = mes[movie1]
        mes2 = mes[movie2]
        if (mes1 and mes2):
            shotLengthDiff = math.fabs(mes1[0] - mes2[0])
            colorVarianceDiff = math.fabs(mes1[1] - mes2[1])
            motionDiff = math.fabs(mes1[3] - mes2[3])
            lightingDiff = math.fabs(mes1[5] - mes2[5])
            numShotsDiff = math.fabs(mes1[6] - mes2[6])
            return shotLengthDiff * colorVarianceDiff * motionDiff * lightingDiff * numShotsDiff
        else:
            return 0

    def estimate(self, u, i):
        # ur(:obj:`defaultdict` of :obj:`list`): The users ratings. This is a
        #     dictionary containing lists of tuples of the form ``(item_inner_id,
        #     rating)``. The keys are user inner ids.
        # ir(:obj:`defaultdict` of :obj:`list`): The items ratings. This is a
        #     dictionary containing lists of tuples of the form ``(user_inner_id,
        #     rating)``. The keys are item inner ids. Item ~ Movie
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        # Build up similarity scores between this item and everything the user rated
        neighbors = []
        # neighbor là 1 list dạng [(độ tg đồng của movie đc chon vs từng movie còn lại,
        # rating của movie còn lại),]
        for rating in self.trainset.ur[u]: # rating co dang (movieId (item innerid), rating)
            genreSimilarity = self.similarities[i,rating[0]] # Trả về giá trị tại 1 vị trí tại ma trận
            neighbors.append((genreSimilarity, rating[1]) )

        
        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])
        
        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
            
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
    