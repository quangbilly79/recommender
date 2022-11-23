import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics:

    def MAE(predictions):
        return accuracy.mae(predictions, verbose=False)

    def RMSE(predictions):
        return accuracy.rmse(predictions, verbose=False)
    # Lấy TopN (dạng Dict) các User và rating tg ứng (Key-Value)
    # từ predictions {user1: [movieId, Rating], user2: [], ...}
    # Có thể thêm cả minimumRating để lọc
    def GetTopN(predictions, n=10, minimumRating=0.0):
        topN = defaultdict(list) # tg tự Dict thg, nhưng có thể thay key rỗng = giá trị default

        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            topN[int(userID)] = ratings[:n]

        return topN

    # Check từng phần tử trong topN của mình (topNPredicted) ~ Trainset
    # và ptu đc bỏ ra từ topN gốc (leftOut) ~ testSet.
    # Nếu trúng thì là hit
    def HitRate(topNPredicted, leftOutPredictions): # dùng left out
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total

    # tương tự cái trên những có thêm điều kiện ratingCutoff
    def CumulativeHitRate(topNPredicted, leftOutPredictions, ratingCutoff=0):
        hits = 0
        total = 0

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Only look at ability to recommend things the users actually liked...
            if (actualRating >= ratingCutoff):
                # Is it in the predicted top 10 for this user?
                hit = False
                for movieID, predictedRating in topNPredicted[int(userID)]:
                    if (int(leftOutMovieID) == movieID):
                        hit = True
                        break
                if (hit) :
                    hits += 1

                total += 1

        # Compute overall precision
        return hits/total
    # Phía trên là ch có 1 biến hit để tính tổng hit trên tất cả user
    # Còn dưới này là tính tổng hít cho từng rating (hit h là 1 dict)
    def RatingHitRate(topNPredicted, leftOutPredictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == movieID):
                    hit = True
                    break
            if (hit) :
                hits[actualRating] += 1

            total[actualRating] += 1

        # Compute overall precision
        for rating in sorted(hits.keys()):
            print (rating, hits[rating] / total[rating])

    # Mỗi khi hit thì lấy rank, 1/rank va tính tổng.
    # rank bđ từ 1, hit càng sớm (rank cao, top 1,2,..) càng tốt
    def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total

    # What percentage of users have at least one "good" recommendation
    # Tổng số hit (movie trong topN Predict mà có điểm Predict > n) / số User
    def UserCoverage(topNPredicted, numUsers, ratingThreshold=0):
        hits = 0
        for userID in topNPredicted.keys():
            hit = False
            for movieID, predictedRating in topNPredicted[userID]:
                if (predictedRating >= ratingThreshold):
                    hit = True
                    break
            if (hit):
                hits += 1

        return hits / numUsers
    # Mức độ đa dạng movie của list Predict (dựa vào sự khác nhau của từng cặp movie trong list Predict)
    def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            # itertools.combinations(list,n): Chia list thành tổ hợp các list nhỏ, mỗi list chứa 2 phần tử k trùng
            # itertools.combinations([1,3,2],2): [1,3], [1,2], [3,2]
            # Chia list các movie trong topN của 1 user bất kì thành 1 cặp 2 movie để tính toán sự khac biệt (Diversity)
            # topN co dang {user1: [movieId, Rating], user2: [0,1], ...}
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                # haàm simsMatrix dung inner id, cần convert movieId to Inner Id (kiểu hash)
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        if (n > 0):
            S = total / n
            return (1-S)
        else:
            return 0
    # Trung bình độ phổ biến của tất cả movie trong list Predict
    def Novelty(topNPredicted, rankings):
        # Ranking là 1 list chứa các moviesID thực te và đc sắp xếp theo độ phổ biến
        n = 0
        total = 0
        for userID in topNPredicted.keys():
            for rating in topNPredicted[userID]:
                movieID = rating[0]
                rank = rankings[movieID] # Lấy rank (độ phổ biến) thực tế của từng movie trong list Predict
                total += rank # Tính tổng rank
                n += 1
        return total / n # Tính trung bình
