#!/usr/bin/env python3
### IMPORTS ###
from __future__ import print_function

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc #note we overwrite native python sum
import pyspark.sql.types as T
from pyspark.sql.types import StringType

import sys


### FUNCTIONS ###


### SPARK INSTANTIATION ###
#TODO: check notes if the spark builder needs adjustments
#spark builder
#remove .master when testing on cloud
#CHECK ANY OTHER CLOUD BUILDERS NEEDED OR NOT
spark = (
    SparkSession.builder
    .appName("toxic_commnt_classifier")
    # .master("local[*]") #DOUBLE CHECK WHAT THIS DOES ON CLOUD
    # .config("spark.driver.bindAddress", "127.0.0.1") #REMOVE ON CLOUD
    .getOrCreate()
)

#To reduce logs outputted CLOUD
sc = spark.sparkContext
sc.setLogLevel("ERROR")  # or "WARN"





### DATA ###
## Load Data

#LOCAL
local_train_path = "../../data/part_1/train.csv"
local_test_path = "../../data/part_1/test.csv"
df_train = spark.read.format("csv").option("header", True).load(local_train_path)
df_test = spark.read.format("csv").option("header", True).load(local_test_path)

#CLOUD COMMAND
#TODO: understand what the sys arguments are going to be here. 
# local_train_path = sys.argv[1]
# local_test_path = sys.argv[1]
# df = spark.read.format("csv").option("header", True).load(df_path)


out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]
