import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('RDD Exercise').getOrCreate()

# Load CSV file into a data frame
score_sheet_df = spark.read.load('/user/zzj/score-sheet.csv', \
    format='csv', sep=';', inferSchema='true', header='true')

score_sheet_df.show()

# Get RDD from the data frame
score_sheet_rdd = score_sheet_df.rdd
score_sheet_rdd.first()

# Project the second column of scores with an additional 1
score_rdd = score_sheet_rdd.map(lambda x: (x[1], 1))
score_rdd.first()

# Get the sum and count by reduce
(sum, count) = score_rdd.reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]))
print('Average Score : ' + str(sum/count))

# Load Parquet file into a data frame
posts_df = spark.read.load('/user/zzj/parquet-input/hardwarezone.parquet')

posts_df.createOrReplaceTempView("posts")
sqlDF = spark.sql("SELECT * FROM posts WHERE author='SG_Jimmy'")
num_post = sqlDF.count()
print('Jimmy has ' + str(num_post) + ' posts.')

posts_rdd = posts_df.rdd

# Project the author and content columns
author_content_rdd = posts_rdd.map(lambda x: (len(x[2]), 1))
author_content_rdd.first()

# Get sume and count by reduce
(sum, count) = author_content_rdd.reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))
print('Average post length : ' + str(sum/count))
