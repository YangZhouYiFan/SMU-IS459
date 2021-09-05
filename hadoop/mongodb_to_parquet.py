import pymongo
import pandas as pd
from pymongo import MongoClient
import pyarrow as pa
import pyarrow.parquet as pq

client = MongoClient('localhost')
db = client['hardwarezone']
posts = db.posts

# extract records from MongoDB without the id field
post_contents = pd.DataFrame(list(posts.find({},{'_id': False})))


# convert to the pyarrow table
table = pa.Table.from_pandas(post_contents)
print("Column names: {}".format(table.column_names))
print("Schema: {}".format(table.schema))

# write to the parquet file
pq.write_table(table, 'hardwarezone.parquet')
