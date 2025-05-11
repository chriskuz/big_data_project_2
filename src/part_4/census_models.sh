#!/bin/bash
source ../../../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /spark-project2/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /spark-project2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/test3.test /spark-project2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/train3.data /spark-project2/input/
/usr/local/spark/bin/spark-submit \
    --master spark://$SPARK_MASTER:7077 \
    ./census_income.py \
    hdfs://$SPARK_MASTER:9000/spark-project2/input/train3.data \
    hdfs://$SPARK_MASTER:9000/spark-project2/input/test3.test