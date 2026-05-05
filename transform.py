import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, min, max, round
from pyspark.sql.types import TimestampType

# 1. Parse arguments passed from the Glue Job parameters
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

# Initialize Apache Iceberg configurations
spark.conf.set("spark.sql.catalog.glue_catalog", "org.apache.iceberg.spark.SparkCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
spark.conf.set("spark.sql.catalog.glue_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
spark.conf.set("spark.sql.catalog.glue_catalog.warehouse", f"s3://{semantic_bucket}/iceberg-warehouse/")
spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

print(f"Reading raw data from: s3://{raw_bucket}/data/")

# 3. Read the raw JSON data
# Spark automatically infers the nested 'payload' object as a StructType
raw_df = spark.read \
    .option("basePath", f"s3://{raw_bucket}/data/") \
    .json(f"s3://{raw_bucket}/data/")

# -----------------------------------------------------------------------------
# Part 1: Data Preparation & JSON Unpacking
# -----------------------------------------------------------------------------

# Since 'payload' is already a Struct, we can flatten it directly with 'payload.*'
parsed_df = raw_df \
    .select(
        col("event_id"),
        col("event_time").cast(TimestampType()).alias("event_time"),
        col("charger_id"),
        col("event_type"),
        col("p_ingest_day"),
        col("payload.*") # Flattens all nested attributes automatically
    )

parsed_df.cache() # Cache the dataframe as we will use it multiple times

# -----------------------------------------------------------------------------
# Part 2: Transformation & Table Creation (Workshop Exercises)
# -----------------------------------------------------------------------------

# --- Table 1: Telemetry History (Fact Table) ---
telemetry_df = parsed_df.filter(col("event_type") == "telemetry") \
    .select(
        "p_ingest_day", "charger_id", "event_time", "connector_id", 
        "status", "temperature_c", "voltage_v", "current_a"
    )

telemetry_df.createOrReplaceTempView("vw_telemetry")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS glue_catalog.{database_name}.telemetry_history (
    p_ingest_day STRING,
    charger_id STRING,
    event_time TIMESTAMP,
    connector_id INT,
    status STRING,
    temperature_c DOUBLE,
    voltage_v DOUBLE,
    current_a DOUBLE
)
USING iceberg
PARTITIONED BY (p_ingest_day, charger_id)
TBLPROPERTIES ('format-version'='2', 'write.delete.mode'='merge-on-read')
""")
spark.sql(f"INSERT OVERWRITE glue_catalog.{database_name}.telemetry_history SELECT * FROM vw_telemetry")
print("Successfully processed telemetry_history")


# --- Table 2: Session Metrics (Aggregated Table) ---
session_df = parsed_df.filter(col("event_type") == "session") \
    .groupBy("p_ingest_day", "session_id", "charger_id", "vehicle_id") \
    .agg(
        min("event_time").alias("session_start_time"),
        max("event_time").alias("session_end_time"),
        max("energy_kwh_total").alias("total_energy_kwh")
    )

session_df.createOrReplaceTempView("vw_session")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS glue_catalog.{database_name}.session_metrics (
    p_ingest_day STRING,
    session_id STRING,
    charger_id STRING,
    vehicle_id STRING,
    session_start_time TIMESTAMP,
    session_end_time TIMESTAMP,
    total_energy_kwh DOUBLE
)
USING iceberg
PARTITIONED BY (p_ingest_day)
TBLPROPERTIES ('format-version'='2', 'write.delete.mode'='merge-on-read')
""")
spark.sql(f"INSERT OVERWRITE glue_catalog.{database_name}.session_metrics SELECT * FROM vw_session")
print("Successfully processed session_metrics")


# --- Table 3: Fault Logs (Filtered View) ---
fault_df = parsed_df.filter(col("event_type") == "fault") \
    .select(
        "p_ingest_day", "charger_id", "event_time", "connector_id",
        "fault_code", "severity", "message"
    )

fault_df.createOrReplaceTempView("vw_faults")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS glue_catalog.{database_name}.fault_logs (
    p_ingest_day STRING,
    charger_id STRING,
    event_time TIMESTAMP,
    connector_id INT,
    fault_code STRING,
    severity STRING,
    message STRING
)
USING iceberg
PARTITIONED BY (p_ingest_day)
TBLPROPERTIES ('format-version'='2', 'write.delete.mode'='merge-on-read')
""")
spark.sql(f"INSERT OVERWRITE glue_catalog.{database_name}.fault_logs SELECT * FROM vw_faults")
print("Successfully processed fault_logs")

# Commit the job
job.commit()
