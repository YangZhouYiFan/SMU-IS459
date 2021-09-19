from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('Neo4jTest').getOrCreate()

# Load data from parquet
posts_df = spark.read.load('/user/zzj/parquet-input/hardwarezone.parquet')


# Clean the data with null values
posts_df = posts_df.na.drop()

posts_df = posts_df.sample(fraction=0.05)

# Build df for distinct authors
author_df = posts_df.select('author') \
    .distinct() \
    .withColumnRenamed('author','name')


print("# of authors " + str(author_df.count()))

# Write to Neo4j as nodes
author_df.write \
    .format('org.neo4j.spark.DataSource') \
    .mode('append') \
    .option('url', 'bolt://localhost:7687') \
    .option('authentication.basic.username', 'neo4j') \
    .option('authentication.basic.password', 'zzj') \
    .option('labels', ':Person:Author') \
    .option('node.keys', 'name') \
    .save()

# Build df for threads
topic_df = posts_df.select('topic') \
    .distinct() \
    .withColumnRenamed('topic','title')

print("# of threads " + str(topic_df.count()))

# Write to Neo4j as nodes
topic_df.write \
    .format('org.neo4j.spark.DataSource') \
    .mode('append') \
    .option('url', 'bolt://localhost:7687') \
    .option('authentication.basic.username', 'neo4j') \
    .option('authentication.basic.password', 'zzj') \
    .option('labels', ':Thread') \
    .option('node.keys', 'title') \
    .save()

# Write to Neo4j as nodes
posts_df.write \
    .format('org.neo4j.spark.DataSource') \
    .mode('append') \
    .option('url', 'bolt://localhost:7687') \
    .option('authentication.basic.username', 'neo4j') \
    .option('authentication.basic.password', 'zzj') \
    .option('labels', ':Post') \
    .option('node.keys', 'author,topic,content') \
    .save()

#df = spark.read.format('org.neo4j.spark.DataSource') \
#    .option('url', 'bolt://localhost:7687') \
#    .option('authentication.basic.username', 'neo4j') \
#    .option('authentication.basic.password', 'zzj') \
#    .option('query', 'MATCH (n:Person) RETURN (n)') \
#    .load()

#num = df.count()
#print(num)
