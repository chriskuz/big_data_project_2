#!/usr/bin/env python3
### IMPORTS ###
from __future__ import print_function

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import ChiSqSelector, VectorAssembler

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc, when #note we overwrite native python sum
import pyspark.sql.types as T
from functools import reduce

import sys


### FUNCTIONS ###



### SPARK INSTANTIATION ###
#spark builder
#remove .master when testing on cloud
#CHECK ANY OTHER CLOUD BUILDERS NEEDED OR NOT
spark = (
    SparkSession.builder
    .appName("diagnose_heart")
    # .master("local[*]") #REMOVE ON CLOUD #bypasses scheduler and uses all cores...good for local dev
    # .config("spark.driver.bindAddress", "127.0.0.1") #REMOVE ON CLOUD #sets driver to avoid any network traffic and all done local
    .getOrCreate()
)

#To reduce logs outputted CLOUD
sc = spark.sparkContext
sc.setLogLevel("ERROR")  # or "WARN"


### DATA ###
## Load Data 

#LOCAL
# local_df_path = "../../data/part_2/framingham.csv"

# df = (
#     spark.read
#     .format("csv")
#     .option("header", True)
#     .option("InferSchema", True)
#     .option("nullvalue", "NA")
#     .load(local_df_path)
# )

#CLOUD 
cloud_df_path = sys.argv[1]
df = (
    spark.read
    .format("csv")
    .option("header", True)
    .option("InferSchema", True)
    .option("nullvalue", "NA")
    .load(cloud_df_path)
)

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






### MODEL ###


feature_cols = [c for c in df.columns if c != "TenYearCHD"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)


## Data Splitting (Post Cleaning)
df_train, df_test = df.randomSplit([0.8, 0.2], seed=20250510)

## Logistic Regression 

#Training
lr = LogisticRegression(featuresCol="features", labelCol="TenYearCHD", maxIter=10, regParam=0.1)
lr_model = lr.fit(df_train)
#Prediction on test
predictions = lr_model.transform(df_test)

## Evaluator

#Model evaluation
evaluator_acc = MulticlassClassificationEvaluator(labelCol="TenYearCHD", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="TenYearCHD", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="TenYearCHD", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="TenYearCHD", metricName="weightedRecall")

print(f"Accuracy: {evaluator_acc.evaluate(predictions):.4f}")
print(f"F1 Score: {evaluator_f1.evaluate(predictions):.4f}")
print(f"Precision: {evaluator_precision.evaluate(predictions):.4f}")
print(f"Recall: {evaluator_recall.evaluate(predictions):.4f}")

#Feature Importance
feature_coeffs = [(name, float(coef)) for name, coef in zip(feature_cols, lr_model.coefficients)] #type conversion

importance_df = spark.createDataFrame(feature_coeffs, ["feature", "coefficient"]) #importance df creation
importance_df = importance_df.withColumn("abs_coefficient", F.abs(col("coefficient"))) #absolute value to showcase influence
importance_df.orderBy(col("abs_coefficient").desc()).show(truncate=False) #ordering and display