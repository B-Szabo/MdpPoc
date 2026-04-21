import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import avg, col, round

# 1. Parse arguments passed from the SAM template
args = getResolvedOptions(sys.argv, [
    'JOB_NAME', 
    'raw_bucket', 
    'semantic_bucket', 
    'database_name'
])

raw_bucket = args['raw_bucket']
semantic_bucket = args['semantic_bucket']
database_name = args['database_name']

# 2. Initialize contexts
sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

spark.conf.set("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
spark.conf.set("spark.sql.catalog.glue_catalog.warehouse", f"s3://{semantic_bucket}/iceberg-warehouse/")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

print(f"Reading raw data from: s3://{raw_bucket}/data/")

# 3. Read the raw JSON data
# We use base path so Spark correctly understands the p_ingest_day partition in the folder structure
raw_df = spark.read \
    .option("basePath", f"s3://{raw_bucket}/data/") \
    .json(f"s3://{raw_bucket}/data/")

# 4. Transform: Calculate average wage per country, per day
# We round the average wage to 2 decimal places for cleaner semantic data
aggregated_df = raw_df.groupBy("p_ingest_day", "country") \
    .agg(round(avg("wage"), 2).alias("avg_wage"))

print("Data aggregated successfully. Sample:")
aggregated_df.show(5)

# 5. Create a temporary view to use standard Spark SQL for Iceberg
aggregated_df.createOrReplaceTempView("daily_aggregates")

table_name = "daily_country_wages"

# Add backticks to escape hyphens in the database name to prevent Spark SQL ParseExceptions
full_table_identifier = f"glue_catalog.{database_name}.{table_name}"
iceberg_location = f"s3://{semantic_bucket}/iceberg_tables/{table_name}/"

# 6. Create the Iceberg table if it doesn't already exist
create_table_query = f"""
CREATE TABLE IF NOT EXISTS {full_table_identifier} (
    p_ingest_day STRING,
    country STRING,
    avg_wage DOUBLE
)
USING iceberg
PARTITIONED BY (p_ingest_day)
LOCATION '{iceberg_location}'
TBLPROPERTIES (
    'format-version'='2',
    'write.delete.mode'='merge-on-read'
)
"""
print(f"Executing create table query: {create_table_query}")
spark.sql(create_table_query)

# 7. Write the data to the Iceberg table
# Using INSERT OVERWRITE with dynamic partition mode ensures we don't duplicate data if re-run
insert_query = f"""
INSERT OVERWRITE {full_table_identifier}
SELECT p_ingest_day, country, avg_wage 
FROM daily_aggregates
"""
print(f"Executing insert overwrite query: {insert_query}")
spark.sql(insert_query)

print(f"Successfully wrote data to Iceberg table: {full_table_identifier}")

# Commit the job
job.commit()