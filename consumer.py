import json
import boto3
import os
import uuid
from datetime import datetime

s3 = boto3.client('s3')
bucket_name = os.environ['RAW_BUCKET_NAME']

def handler(event, context):
    records = []
    today = datetime.utcnow().strftime('%Y-%m-%d')

    for record in event['Records']:
        # Athena JSON Lines expects newline delimited objects, not a single array
        body = record['body'] 
        records.append(body)

    if records:
        # Join with newline character
        file_content = "\n".join(records)
        file_key = f"data/p_ingest_day={today}/{uuid.uuid4()}.json"

        s3.put_object(
            Bucket=bucket_name,
            Key=file_key,
            Body=file_content
        )
        print(f"Wrote {len(records)} records to s3://{bucket_name}/{file_key}")

    return {"statusCode": 200, "body": f"Processed {len(records)} records"}
