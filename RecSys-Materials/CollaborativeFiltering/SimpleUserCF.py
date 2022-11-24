# -*- coding: utf-8 -*-
"""
Created on Wed May  9 10:10:04 2018

@author: Frank
"""
# Ngoài dùng để lay' top N, module nay` con` du, doan' rating
# Khong chia lam` 2 set test vi` k can` tinh' accuracy
from MovieLens import MovieLens
from surprise import KNNBasic
import heapq
from collections import defaultdict
from operator import itemgetter
        
testSubject = '85' # Ss user id 85 voi cac user khac de recommend cho user nay
k = 10

# Load our data set and compute the user similarity matrix giua~ 1 cap. 2 user bat' ki`
ml = MovieLens()
data = ml.loadMovieLensLatestSmall()

trainSet = data.build_full_trainset()

sim_options = {'name': 'cosine',
               'user_based': True
               }
# KNNBasic Class: def __init__(self,
#              k: int = 40,
#              min_k: int = 1,
#              sim_options: dict = {},
#              verbose: bool = True,
#              **kwargs: Any) -> None
model = KNNBasic(sim_options=sim_options)
# fit: This method is called by every derived class as the first basic step for training an algorithm.
# It basically just initializes some internal structures and set the self.trainset attribute.
model.fit(trainSet)
# Ham` compute_similarities(): Build the similarity matrix.
# The way the similarity matrix is computed depends on the sim_options parameter passed at the creation of the algorithm
# (see similarity_measures_configuration).
# This method is only relevant for algorithms using a similarity measure, such as the k-NN algorithms  .
simsMatrix = model.compute_similarities()

# Get top N similar users to our test subject
# (Alternate approach would be to select users up to some similarity threshold - try it!)
testUserInnerID = trainSet.to_inner_uid(testSubject)
similarityRow = simsMatrix[testUserInnerID] #user-user sim matrix

similarUsers = [] # Lay' tat ca? user tuong dong` vs user dc chon. (user 85)
for innerID, score in enumerate(similarityRow): # Lay' userId va simScore vs user 85
    if (innerID != testUserInnerID):
        similarUsers.append( (innerID, score) )

kNeighbors = heapq.nlargest(k, similarUsers, key=lambda t: t[1])# Lay' top k similarity
# Tim` theo gia tri rating toi thieu
# kNeighbors = []
# for rating in similarUsers:
#     if rating[1] > 0.95:
#         kNeighbors.append(rating)

# Get the stuff they rated, and add up ratings for each item, weighted by user similarity
candidates = defaultdict(float)
for similarUser in kNeighbors:
    innerID = similarUser[0]
    userSimilarityScore = similarUser[1]
    theirRatings = trainSet.ur[innerID] # (innerId, rating)
    for rating in theirRatings:
        candidates[rating[0]] += (rating[1] / 5.0) * userSimilarityScore
    
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



