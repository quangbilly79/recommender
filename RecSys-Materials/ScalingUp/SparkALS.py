# -*- coding: utf-8 -*-
"""
Created on Mon May 28 11:09:55 2018

@author: Frank
"""

from pyspark.sql import SparkSession

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS # Alternating Least Squares matrix factorization
from pyspark.sql import Row

from MovieLens import MovieLens

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ALSExample")\
        .getOrCreate()

    lines = spark.read.option("header", "true").csv("../../ml-latest-small/ratings.csv").rdd
    # Tạo Schema, tạo rdd rồi chuyển sang data frame
    ratingsRDD = lines.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2]), timestamp=int(p[3])))
    
    ratings = spark.createDataFrame(ratingsRDD)
    
    (training, test) = ratings.randomSplit([0.8, 0.2]) # Chia thành training set và test set

    # Pass các giá trị theo yêu câ
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training) # Fits a model to the input dataset Returns Transformer or a list of Transformer
    # ALSModel: uid=ALS_4adcc5c5cde7, rank=10
    # Áp model cho 1 dataset để dùng các hàm khác, hiện tại thì thấy có thêm cột Prediction

    predictions = model.transform(test) # DataFrame[userId: bigint, movieId,.... prediction: float]. Có thêm cột prediction

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    # Tính rmse từ testset = cách ss giữa predict rating và test rating.
    # Ngoài ra có thể thay rmse trong metricName bên trên = mse, mae,...
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    userRecs = model.recommendForAllUsers(10) # recommend top N(10) cho mng
    # DataFrame[userId: int, recommendations: array<struct<movieId:int,rating:float>>]
    user85Recs = userRecs.filter(userRecs['userId'] == 85).collect() # recommend top 10 cho userID 85
    
    spark.stop()

    ml = MovieLens()
    ml.loadMovieLensLatestSmall()

    for row in user85Recs:
        for rec in row.recommendations:
            print(ml.getMovieName(rec.movieId))

