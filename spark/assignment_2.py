import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from graphframes import *

spark = SparkSession.builder.appName('sg.edu.smu.is459.assignment2').getOrCreate()

# Load data
posts_df = spark.read.load('/user/zzj/parquet-input/hardwarezone.parquet')

# Clean the dataframe by removing rows with any null value
posts_df = posts_df.na.drop()

#posts_df.createOrReplaceTempView("posts")

# Find distinct users
#distinct_author = spark.sql("SELECT DISTINCT author FROM posts")
author_df = posts_df.select('author').distinct()

print('Author number :' + str(author_df.count()))

# Assign ID to the users
author_id = author_df.withColumn('id', monotonically_increasing_id())
author_id.show()

# Construct connection between post and author
left_df = posts_df.select('topic', 'author') \
    .withColumnRenamed("topic","ltopic") \
    .withColumnRenamed("author","src_author")

right_df =  left_df.withColumnRenamed('ltopic', 'rtopic') \
    .withColumnRenamed('src_author', 'dst_author')

#  Self join on topic to build connection between authors
author_to_author = left_df. \
    join(right_df, left_df.ltopic == right_df.rtopic) \
    .select(left_df.src_author, right_df.dst_author) \
    .distinct()
edge_num = author_to_author.count()
print('Number of edges with duplicate : ' + str(edge_num))

# Convert it into ids
id_to_author = author_to_author \
    .join(author_id, author_to_author.src_author == author_id.author) \
    .select(author_to_author.dst_author, author_id.id) \
    .withColumnRenamed('id','src')

id_to_id = id_to_author \
    .join(author_id, id_to_author.dst_author == author_id.author) \
    .select(id_to_author.src, author_id.id) \
    .withColumnRenamed('id', 'dst')

id_to_id = id_to_id.filter(id_to_id.src >= id_to_id.dst).distinct()

id_to_id.cache()

print("Number of edges without duplciate :" + str(id_to_id.count()))

# Build graph with RDDs
graph = GraphFrame(author_id, id_to_id)

# For complex graph queries, e.g., connected components, you need to set
# the checkopoint directory on HDFS, so Spark can handle failures.
# Remember to change to a valid directory in your HDFS
spark.sparkContext.setCheckpointDir('/user/zzj/spark-checkpoint')

# The rest is your work, guys
# ......


# Display the vertex and edge DataFrames
graph.vertices.show()

graph.edges.show()

# get author id with combined components
author_component = graph.connectedComponents()
graph.connectedComponents().show()

# aggregate by component
component = graph.connectedComponents().groupBy("component").count()
component.show()


# Q1: get number of communities sort by number of authors within the community in descending order
from pyspark.sql.functions import sum, col, desc
component.sort(desc("count")).show()

component_to_topic = author_component.join(posts_df, author_component.author == posts_df.author).select(author_component.component, posts_df.topic).distinct()
component_to_topic.show()


#data cleaning and preprocessing for generating bigram keywords
component_to_topic.topic = lower(component_to_topic.topic)
component_to_topic.topic = regexp_replace(component_to_topic.topic, "^rt ", "")
component_to_topic.topic = regexp_replace(component_to_topic.topic, "(https?\://)\S+", "")
component_to_topic.topic = regexp_replace(component_to_topic.topic, "[^a-zA-Z0-9\\s]", "")


#tokenize
from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol="topic", outputCol="vector")
vector_df = tokenizer.transform(component_to_topic).select("vector")


# remove stopwords
from pyspark.ml.feature import StopWordsRemover
remover = StopWordsRemover()
stopwords = remover.getStopWords() 


# Specify input/output columns
remover.setInputCol("vector")
remover.setOutputCol("vector_no_stopw")


# Transform existing dataframe with the StopWordsRemover
vector_no_stopw_df = remover.transform(vector_df).select("vector_no_stopw")

vector_no_stopw_df.show()


# get bigram phrases
from pyspark.ml.feature import NGram
ngram = NGram(n=2, inputCol="vector_no_stopw", outputCol="bigrams")
vector_no_stopw_df = ngram.transform(vector_no_stopw_df)

# get word count of bigram phrases in descending order
from pyspark.sql.functions import explode
vector_no_stopw_df.select(explode("bigrams").alias("bigram")).groupBy("bigram").count().sort(desc("count")).show()


#-------------------------------------------------------------------------------------------------------------------

#Q2 get number of triangles in component 0
component_0_author_id = author_id.join(author_component.filter(author_component.component==0), author_component.author==author_id.author).select(author_id.author, author_id.id).distinct()
component_0_author_id.show()
component_0_author_id.count()


component_0_id_to_id = component_0_author_id.join(id_to_id, component_0_author_id.id==id_to_id.src).select(id_to_id.src, id_to_id.dst)
component_0_id_to_id.show()
component_0_id_to_id.count()


component_0_graph = GraphFrame(component_0_author_id, component_0_id_to_id)

component_0_triangles = component_0_graph.triangleCount()


component_0_triangles.sort(desc("count")).show()


from pyspark.sql.functions import *
from pyspark.sql.types import *
component_0_triangles.select(avg("count")).show()
