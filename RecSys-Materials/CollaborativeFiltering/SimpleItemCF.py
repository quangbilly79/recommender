# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:10:04 2018

@author: Frank
"""

from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
        
testSubject = '85' # Ss user id 85 voi cac user khac de recommend cho user nay
k = 10

ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': False
               }

model = KNNBasic(sim_options=sim_options)
model.fit(trainSet)
simsMatrix = model.compute_similarities()

testUserInnerID = trainSet.to_inner_uid(testSubject)

# Get the top K items we rated
testUserRatings = trainSet.ur[testUserInnerID]
# testUserRatings [(650, 5.0), (319, 2.0), (651, 3.0),...] (movieId (ItemId), rating)

# Tìm theo top k rating
kNeighbors = heapq.nlargest(k, testUserRatings, key=lambda t: t[1])
# Tim` theo gia tri rating toi thieu
# kNeighbors = []
# for rating in testUserRatings:
#     if rating[1] > 4.0:
#         kNeighbors.append(rating)

# Get similar items to stuff we liked (weighted by rating)
candidates = defaultdict(float)
for itemID, rating in kNeighbors:
    similarityRow = simsMatrix[itemID] # item-item sim matrix
    #  [0.95106394 0.95850338 0.97227442 ...]
    for innerID, score in enumerate(similarityRow):
        candidates[innerID] += score * (rating / 5.0)

# Build a dictionary of stuff the user has already seen
watched = {}
for itemID, rating in trainSet.ur[testUserInnerID]:
    watched[itemID] = 1
    
# Get top-rated items from similar users:
pos = 0
for itemID, ratingSum in sorted(candidates.items(), key=itemgetter(1), reverse=True):
    if not itemID in watched:
        movieID = trainSet.to_raw_iid(itemID)
        print(ml.getMovieName(int(movieID)), ratingSum)
        pos += 1
        if (pos > 10):
            break

    


