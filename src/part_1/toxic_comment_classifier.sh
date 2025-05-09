#!/bin/bash
source ../../../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /spark-project2/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /spark-project2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/test.csv /spark-project2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/train.csv /spark-project2/input/
/usr/local/spark/bin/spark-submit \
    --master spark://$SPARK_MASTER:7077 \
    ./toxic_comment_classifier.py \
    hdfs://$SPARK_MASTER:9000/spark-project2/input/train.csv \
    hdfs://$SPARK_MASTER:9000/spark-project2/input/test.csv