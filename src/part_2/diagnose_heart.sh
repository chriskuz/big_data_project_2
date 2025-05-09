#!/bin/bash
source ../../../../env.sh
/usr/local/hadoop/bin/hdfs dfs -rm -r /spark-project2/input/
/usr/local/hadoop/bin/hdfs dfs -mkdir -p /spark-project2/input/
/usr/local/hadoop/bin/hdfs dfs -copyFromLocal ../../data/framingham.csv /spark-project2/input/
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 ./diagnose_heart.py hdfs://$SPARK_MASTER:9000/spark-project2/input/