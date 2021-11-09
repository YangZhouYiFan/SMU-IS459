from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import explode, regexp_replace
from pyspark.sql.functions import split
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf, from_json, col, window
from pyspark.sql.types import StringType, StructType, StructField, ArrayType
from pyspark.sql import functions as F
import re
from pyspark.ml.feature import Tokenizer, StopWordsRemover


def parse_data_from_kafka_message(sdf, schema):
    from pyspark.sql.functions import split
    assert sdf.isStreaming == True, "DataFrame doesn't receive streaming data"
    col = split(sdf['value'], ',')

    #split attributes to nested array in one Column
    #now expand col to multiple top-level columns
    for idx, field in enumerate(schema):
        sdf = sdf.withColumn(field.name, col.getItem(idx).cast(field.dataType))
    return sdf

def get_top_10_authors(df, epochId):
    df \
        .orderBy(col("window").desc(), col("count").desc()) \
        .filter(col('author').rlike("[^ +]")) \
        .limit(10) \
        .show(truncate=False)

def get_top_10_words(df, epochId):
    df \
        .orderBy(col("window").desc(), col("count").desc()) \
        .filter(col('word').rlike("[^ +]")) \
        .limit(10) \
        .show(truncate=False)

## data processing function to remove meaningless text in author and content columns
def preprocessing(lines):
    lines = regexp_replace(lines, r'[^A-Za-z\n ]|(http\S+)|(www.\S+)', '')
    lines = regexp_replace(lines, '@\w+', '')
    lines = regexp_replace(lines, '#', '')
    lines = regexp_replace(lines, 'RT', '')
    lines = regexp_replace(lines, ':', '')
    lines = regexp_replace(lines, 'content', '')
    lines = regexp_replace(lines, 'author', '')

    return lines

if __name__ == "__main__":
    # specify the schema of the fields
    hardwarezoneSchema = StructType([ \
        StructField("topic", StringType()), \
        StructField("author", StringType()), \
        StructField("content", StringType()) \
        ])

    # set window and trigger interval
    window_interval = "2 minutes"
    trigger_interval = "1 minutes"

    # set watermark
    watermark_time = "2 minutes"

    spark = SparkSession.builder \
               .appName("KafkaWordCount") \
               .getOrCreate()

    # from Kafka's topic scrapy-output
    df = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:9092") \
            .option("subscribe", "scrapy-output") \
            .option("startingOffsets", "earliest") \
            .load() \
            .selectExpr("CAST(value AS STRING)", "timestamp") \
            .dropna()

    # get top 10 authors by count
    top_authors = parse_data_from_kafka_message(df, hardwarezoneSchema) \
        .select("author", "timestamp") \
        .select(preprocessing("author").alias("author"), "timestamp") \
        .withWatermark("timestamp", watermark_time) \
        .groupBy(
            window("timestamp", window_interval, trigger_interval), "author").count() \
        .writeStream \
        .outputMode('append') \
        .foreachBatch(get_top_10_authors) \
        .start()

    # set stopwords remover
    stopwords = StopWordsRemover()
    stopwords.setInputCol("word")
    stopwords.setOutputCol("word_cleaned")

    # set tokenizer to transform content column
    tokenizer = Tokenizer(inputCol="content_cleaned", outputCol="word")


    lines = parse_data_from_kafka_message(df, hardwarezoneSchema) \
            .select("content", "timestamp")
            
    lines_cleaned = lines.select(preprocessing("content").alias("content_cleaned"), "timestamp")

    words = tokenizer.transform(lines_cleaned).select("word", "timestamp")
    words_cleaned = stopwords.transform(words).select("word_cleaned", "timestamp")

    # get top 10 words by count
    top_words = words_cleaned \
        .select("word_cleaned", "timestamp") \
        .select(explode("word_cleaned").alias('word'), "timestamp") \
        .withWatermark("timestamp", watermark_time) \
        .groupBy(
            window("timestamp", window_interval, trigger_interval), "word").count() \
        .writeStream \
        .outputMode('append') \
        .foreachBatch(get_top_10_words) \
        .start()
    

    top_authors.awaitTermination()
    top_words.awaitTermination()
    

    