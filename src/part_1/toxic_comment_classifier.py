#!/usr/bin/env python3
### IMPORTS ###
from __future__ import print_function

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, countDistinct, sum, count, mean, isnan, asc, desc #note we overwrite native python sum
import pyspark.sql.types as T
from pyspark.sql.types import StringType, FloatType

import sys


### FUNCTIONS ###
def prep_engineering(df): #no need for a udf here
    #split words
    tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
    tokenized_words = tokenizer.transform(df)

    #
    hased_term_freq = HashingTF(inputCol="words", outputCol="raw_features")
    tf = hased_term_freq.transform(tokenized_words)

    idf = IDF(inputCol="raw_features", outputCol="features")
    idfModel = idf.fit(tf)
    tfidf = idfModel.transform(tf)

    tfidf.select("features").first()

    return tfidf

def classify_me(df, label_col, reg_param, line_limit=5000):
    lr = LogisticRegression(featuresCol="features", labelCol=label_col, regParam=reg_param)

    lr_model = lr.fit(df.limit(line_limit))

    df_results = lr_model.transform(df)

    return df_results


probability_extraction = udf(lambda x: float(x[1]), FloatType())


### SPARK INSTANTIATION ###
#TODO: check notes if the spark builder needs adjustments
#spark builder
#remove .master when testing on cloud
#CHECK ANY OTHER CLOUD BUILDERS NEEDED OR NOT
spark = (
    SparkSession.builder
    .appName("toxic_commnt_classifier")
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
local_train_path = "../../data/part_1/train.csv"
local_test_path = "../../data/part_1/test.csv"
df_train = (
    spark.read
    .format("csv")
    .option("header", True)
    .option("multiline", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("inferSchema", True)
    .load(local_train_path)
)
df_test = (
    spark.read
    .format("csv")
    .option("header", True)
    .option("multiline", True)
    .option("quote", '"')
    .option("escape", '"')
    .option("inferSchema", True)
    .load(local_test_path)
)

#CLOUD
#TODO: understand what the sys arguments are going to be here. ...also copy the format as shown above
# local_train_path = sys.argv[1]
# local_test_path = sys.argv[1]
# df = spark.read.format("csv").option("header", True).load(df_path)


# out_cols = [i for i in df_train.columns if i not in ["id", "comment_text"]] #this is for saving

# df_train, df_test = df.randomSplit([0.8, 0.2], seed=20250511) #we just show how we can split the data

toxic_col_relationship = (
    df_train
    .where(
        (
            (col("toxic") == 0) &
            (
                (col("severe_toxic") == 1) |
                (col("obscene") == 1) |
                (col("threat") == 1) |
                (col("insult") == 1) |
                (col("identity_hate") == 1)
            )
        )
    )
)
print("\n Showing the relationship of toxic against other calssifications. Note that there isn't total coincidence, verifying that a blank 'toxic' term is not applied to every toxic comment. \n")
toxic_col_relationship.show()

### MODEL ###


## Model Prep
transformed_df_train = prep_engineering(df_train)
transformed_df_test = prep_engineering(df_test)

transformed_df_train.show()
transformed_df_test.show()



## Main Model Instantiation
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
result_dfs = []
reg_param = 1
#df_train model gen
for label in labels:
    df_result = classify_me(transformed_df_train, label, reg_param, 5000)
    print(f"\nShowing training sample model output for {label} column probability....\n")
    # df_result.select("id", label, "probability", "prediction").show(10)
    df_result.withColumn("extracted_probability", probability_extraction("probability")).select("comment_text", label, "extracted_probability", "prediction").show(40)
    result_dfs.append(df_result)
