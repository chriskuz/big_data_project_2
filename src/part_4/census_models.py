#!/usr/bin/env python3
### IMPORTS ###
from __future__ import print_function

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import ChiSqSelector, VectorAssembler
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc, when #note we overwrite native python sum
import pyspark.sql.types as T
from functools import reduce

import sys

### SPARK INSTANTIATION ###
#spark builder
#remove .master when testing on cloud
#CHECK ANY OTHER CLOUD BUILDERS NEEDED OR NOT
spark = (
    SparkSession.builder
    .appName("cencus_income")
    #.master("local[*]") #REMOVE ON CLOUD #bypasses scheduler and uses all cores...good for local dev
    #.config("spark.driver.bindAddress", "127.0.0.1") #REMOVE ON CLOUD #sets driver to avoid any network traffic and all done local
    .getOrCreate()
)

#To reduce logs outputted CLOUD
sc = spark.sparkContext
sc.setLogLevel("ERROR")  # or "WARN"


### DATA ###
## Load Data

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

##LOCAL
#local_train_path = "../../data/train3.data"
#local_test_path = "../../data/test3.test"
#df_train = (
#    spark.read
#    .format("csv")
#    .option("header", False)
#    .option("inferSchema", True)
#    .load(local_train_path)
#)
#df_train = df_train.toDF(*columns)

#df_test = (
#    spark.read
#    .format("csv")
#    .option("header", False)
#    .option("inferSchema", True)
#    .option("comment", "|")
#    .load(local_test_path)
#)
#df_test = df_test.toDF(*columns)

#CLOUD
#TODO: understand what the sys arguments are going to be here. ...also copy the format as shown above
train_path = sys.argv[1]
test_path = sys.argv[2] 
df_train = (
    spark.read
    .format("csv")
    .option("header", False)
    .option("inferSchema", True)
    .load(train_path)
)
df_train = df_train.toDF(*columns)

df_test = (
    spark.read
    .format("csv")
    .option("header", False)
    .option("inferSchema", True)
    .option("comment", "|")
    .load(test_path)
)
df_test = df_test.toDF(*columns)


# Null Analysis and Cleaning for df_train

# total row count
num_rows = df_train.count()
print(f"\nNumber of rows in Adult Income Training Dataset: {num_rows}\n")

# nulls per column
null_exprs = [sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in df_train.columns]
null_counts_df = df_train.select(null_exprs)
null_counts_df.show()

# total nulls
total_nulls_expr = reduce(lambda a, b: a + b, [col(c) for c in null_counts_df.columns])
total_nulls = null_counts_df.select(total_nulls_expr.alias("total_nulls")).first()["total_nulls"]
print(f"Total nulls: {total_nulls}")
print(f"The max possible null composition of the data: {total_nulls / num_rows:.4f}")

categorical_cols = ["workclass", "occupation", "native_country"]

for col_name in categorical_cols:
    count = df_train.filter(F.col(col_name) == "?").count()
    print(f'Missing entries ("?") in {col_name}: {count}')


# Null Analysis and Cleaning for df_test

# total row count
num_rows = df_test.count()
print(f"\nNumber of rows in Adult Income Testing Dataset: {num_rows}\n")

# nulls per column
null_exprs = [sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in df_test.columns]
null_counts_df = df_test.select(null_exprs)
null_counts_df.show()

# total nulls
total_nulls_expr = reduce(lambda a, b: a + b, [col(c) for c in null_counts_df.columns])
total_nulls = null_counts_df.select(total_nulls_expr.alias("total_nulls")).first()["total_nulls"]
print(f"Total nulls: {total_nulls}")
print(f"The max possible null composition of the data: {total_nulls / num_rows:.4f}")

categorical_cols = ["workclass", "occupation", "native_country"]

for col_name in categorical_cols:
    count = df_test.filter(F.col(col_name) == "?").count()
    print(f'Missing entries ("?") in {col_name}: {count}')


# apparently there are no nulls in the test set or the training set, so we don't need to drop any rows


# handling categorical variables
categorical_cols = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country"
]

# Index categorical columns
indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep")
    for col in categorical_cols
]

# One-hot encode the indexed columns
encoders = [
    OneHotEncoder(inputCol=col + "_index", outputCol=col + "_encoded")
    for col in categorical_cols
]

# Define numeric columns
numeric_cols = [
    "age", "fnlwgt", "education_num", "capital_gain",
    "capital_loss", "hours_per_week"
]

# Combine all feature columns
feature_cols = [col + "_encoded" for col in categorical_cols] + numeric_cols

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Create label indexer
label_indexer = StringIndexer(inputCol="income", outputCol="label", handleInvalid="keep")


# Clean up trailing period in income values
df_train = df_train.withColumn("income", F.regexp_replace("income", r"\.", ""))
df_test = df_test.withColumn("income", F.regexp_replace("income", r"\.", ""))


# Create the pipeline
stages = indexers + encoders + [label_indexer, assembler]

pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(df_train)
df_train_transformed = pipeline_model.transform(df_train)
df_test_transformed = pipeline_model.transform(df_test)

#Now df_train_transformed contains:
#features: your ML-ready feature vector
#label: the indexed binary label

##EVALUATION##

# Evaluate on the predicted DataFrame
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

## --- MODEL: RANDOM FOREST ---
print("\n===== Random Forest Model =====\n")

rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100, maxDepth=10)
rf_model = rf.fit(df_train_transformed)
rf_predictions = rf_model.transform(df_test_transformed)

print("=== Random Forest Results ===")
rf_predictions.select("income", "label", "prediction", "probability").show(10, truncate=False)

print("Random Forest Evaluation:")
print(f"Accuracy:  {evaluator_acc.evaluate(rf_predictions):.4f}")
print(f"F1 Score:  {evaluator_f1.evaluate(rf_predictions):.4f}")
print(f"Precision: {evaluator_precision.evaluate(rf_predictions):.4f}")
print(f"Recall:    {evaluator_recall.evaluate(rf_predictions):.4f}")



##MODEL - DECISION TREE##

## --- MODEL: DECISION TREE ---
print("\n===== Decision Tree Model =====\n")

dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", maxDepth=10)
dt_model = dt.fit(df_train_transformed)
dt_predictions = dt_model.transform(df_test_transformed)

print("=== Decision Tree Results ===")
dt_predictions.select("income", "label", "prediction", "probability").show(10, truncate=False)

print("Decision Tree Evaluation:")
print(f"Accuracy:  {evaluator_acc.evaluate(dt_predictions):.4f}")
print(f"F1 Score:  {evaluator_f1.evaluate(dt_predictions):.4f}")
print(f"Precision: {evaluator_precision.evaluate(dt_predictions):.4f}")
print(f"Recall:    {evaluator_recall.evaluate(dt_predictions):.4f}")