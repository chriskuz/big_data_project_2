#!/usr/bin/env python3
### IMPORTS ###
from __future__ import print_function

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import ChiSqSelector

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc, when #note we overwrite native python sum
import pyspark.sql.types as T
from functools import reduce

import sys


### FUNCTIONS ###




### SPARK INSTANTIATION ###
#TODO: check notes if the spark builder needs adjustments
#spark builder
#remove .master when testing on cloud
#CHECK ANY OTHER CLOUD BUILDERS NEEDED OR NOT
spark = (
    SparkSession.builder
    .appName("diagnose_heart")
    .master("local[*]") #REMOVE ON CLOUD #bypasses scheduler and uses all cores...good for local dev
    # .config("spark.driver.bindAddress", "127.0.0.1") #REMOVE ON CLOUD #sets driver to avoid any network traffic and all done local
    .getOrCreate()
)

#To reduce logs outputted CLOUD
sc = spark.sparkContext
sc.setLogLevel("ERROR")  # or "WARN"


### DATA ###
## Load Data 

#LOCAL
local_df_path = "../../data/part_2/framingham.csv"

df = (
    spark.read
    .format("csv")
    .option("header", True)
    .option("InferSchema", True)
    .option("nullvalue", "NA")
    .load(local_df_path)
)

#CLOUD 
# cloud_df_path = sys.argv[1]
# df = (
#     spark.read
#     .format("csv")
#     .option("header", True)
#     .option("InferSchema", True)
#     .option("nullvalue", "NA")
#     .load(cloud_df_path)
# )


#showcases data types
df.printSchema()

## Cleaning

# Null Analysis and Cleaning

#gets total rows
num_rows = df.count()
print(f"\nNumber of rows in Framingham Dataset: {num_rows}\n")

#nulls per column counts
null_exprs = [sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns]
null_counts_df = df.select(null_exprs)
null_counts_df.show()

#sum all the nulls from the null_df
total_nulls_expr = reduce(lambda a, b: a + b, [col(c) for c in null_counts_df.columns])
total_nulls = null_counts_df.select(total_nulls_expr.alias("total_nulls")).first()["total_nulls"]
print(f"Total nulls: {total_nulls}")
print(f"The max possible null composition of the data: {total_nulls / num_rows}")

#drops nulls
df = df.dropna()
purified_num_rows = df.count()
print(f"The new row count of the purified dataframe: {purified_num_rows} ")
print(f"The purified dataframe's composition of the original: {purified_num_rows/num_rows}")

#LASSO USE BEST FEATURES
# lr = LogisticRegression(featuresCol="features", labelCol="label", 
#                         elasticNetParam=1.0, regParam=0.1)
# model = lr.fit(data)
# model.coefficients  # Sparse vector showing non-zero coefficients


#CHI SQUARED SELECTOR
# selector = ChiSqSelector(numTopFeatures=50, 
#                          featuresCol="features", 
#                          outputCol="selectedFeatures", 
#                          labelCol="label")

# result = selector.fit(data).transform(data)