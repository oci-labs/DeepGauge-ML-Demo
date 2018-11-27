# Cloud Functions
## Background Cloud Function
Writing, deploying, and triggering a Background Cloud Function with a Cloud Storage trigger.

**Objectives**
- Write and deploy a Background Cloud Function.
- Trigger the function by uploading a file to Cloud Storage.

Update and install gcloud components:
```
gcloud components update &&
gcloud components install beta
```

### Preparing the application
1. Create a Cloud Storage bucket to upload a test file, where YOUR_TRIGGER_BUCKET_NAME is a globally unique bucket name:
```
gsutil mb gs://YOUR_TRIGGER_BUCKET_NAME
gsutil mb gs://ocideepgauge-images
```

### Deploying and triggering the function


Currently, Cloud Storage functions are based on [Pub/Sub notifications from Cloud Storage](https://cloud.google.com/storage/docs/pubsub-notifications) and support similar event types:

-   [finalize](https://cloud.google.com/functions/docs/tutorials/storage#object_finalize)

-   [delete](https://cloud.google.com/functions/docs/tutorials/storage#object_delete)

-   [archive](https://cloud.google.com/functions/docs/tutorials/storage#object_archive)

-   [metadata update](https://cloud.google.com/functions/docs/tutorials/storage#object_metadata_update)

The following sections describe how to deploy and trigger a function for each of these event types.

### Object Finalize

Object finalize events trigger when a "write" of a [Cloud Storage Object](https://cloud.google.com/storage/docs/json_api/v1/objects#resource) is successfully finalized. In particular, this means that creating a new object or overwriting an existing object triggers this event. Archive and metadata update operations are ignored by this trigger.

#### Object Finalize: deploying the function

Take a look at the sample function, which handles Cloud Storage events:
```
def hello_gcs_generic(data, context):
    """Background Cloud Function to be triggered by Cloud Storage.
       This generic function logs relevant data when a file is changed.

    Args:
        data (dict): The Cloud Functions event payload.
        context (google.cloud.functions.Context): Metadata of triggering event.
    Returns:
        None; the output is written to Stackdriver Logging
    """

    print('Event ID: {}'.format(context.event_id))
    print('Event type: {}'.format(context.event_type))
    print('Bucket: {}'.format(data['bucket']))
    print('File: {}'.format(data['name']))
    print('Metageneration: {}'.format(data['metageneration']))
    print('Created: {}'.format(data['timeCreated']))
    print('Updated: {}'.format(data['updated']))
```
Cloud Functions looks for your code in a file named `main.py`.

To deploy the function, run the following command in the directory where the sample code is located:

```
gcloud functions deploy hello_gcs_generic --runtime python37 --trigger-resource YOUR_TRIGGER_BUCKET_NAME --trigger-event google.storage.object.finalize
```
Deep Gauge
```
gcloud functions deploy predict_gauge --source=. --runtime python37 --trigger-resource ocideepgauge-images --trigger-event google.storage.object.metadataUpdate
```

#### Object Finalize: triggering the function

To trigger the function:

1.  Create an empty `gcf-test.txt` file in the directory where the sample code is located.

2.  Upload the file to Cloud Storage in order to trigger the function:
```
  gsutil cp gcf-test.txt gs://ocideepgauge-images
  gsutil cp daisy.jpg gs://ocideepgauge-images
```


3.  Check the logs to make sure the executions have completed:
```
gcloud functions logs read --limit 50
```

### Object Delete

Object delete events are most useful for [non-versioning buckets](https://cloud.google.com/storage/docs/object-versioning). They are triggered when an old version of an object is deleted. In addition, they are triggered when an object is overwritten. Object delete triggers can also be used with [versioning buckets](https://cloud.google.com/storage/docs/object-versioning), triggering when a version of an object is permanently deleted.

#### Object Delete: deploying the function

Using the same sample code as in the finalize example, deploy the function with object delete as the trigger event. Run the following command in the directory where the sample code is located:

```
gcloud functions deploy hello_gcs_generic --runtime python37 --trigger-resource YOUR_TRIGGER_BUCKET_NAME --trigger-event google.storage.object.delete
```

```
gcloud functions deploy predict_gauge --runtime python37 --trigger-resource ocideepgauge-images --trigger-event google.storage.object.delete
```

where `*YOUR_TRIGGER_BUCKET_NAME*` is the name of the Cloud Storage bucket that triggers the function.

#### Object Delete: triggering the function

To trigger the function:

1.  Create an empty `gcf-test.txt` file in the directory where the sample code is located.

2.  Make sure that your bucket is non-versioning:

    gsutil versioning set off gs://*YOUR_TRIGGER_BUCKET_NAME*

3.  Upload the file to Cloud Storage:

    gsutil cp gcf-test.txt gs://*YOUR_TRIGGER_BUCKET_NAME*

    where `*YOUR_TRIGGER_BUCKET_NAME*` is the name of your Cloud Storage bucket where you will upload a test file. At this point the function should not execute yet.

4.  Delete the file to trigger the function:

    gsutil rm gs://*YOUR_TRIGGER_BUCKET_NAME*/gcf-test.txt

5.  Check the logs to make sure the executions have completed:

    `gcloud functions logs read --limit 50\
    `

Note that the function may take some time to finish executing.

# PubSub

## Create a topic and a subscription

Once you create a topic, you can subscribe or publish to it.

Use the [gcloud pubsub topics create](https://cloud.google.com/sdk/gcloud/reference/pubsub/topics/create) command to create a topic:
```
gcloud pubsub topics create my-topic
```
Use the [gcloud pubsub subscriptions create](https://cloud.google.com/sdk/gcloud/reference/pubsub/subscriptions/create) command to create a subscription. Only messages published to the topic after the subscription is created are available to subscriber applications.
```
gcloud pubsub subscriptions create my-sub --topic my-topic
gcloud pubsub subscriptions create gauge-prediction --topic gauge-prediction
```
# BigQuery

## Create BigQuery Dataset and Table

i)Creating a dataset
```
from google.cloud import bigquery
client = bigquery.Client()
dataset_id = 'my_dataset'
dataset_ref = client.dataset(dataset_id)
dataset = bigquery.Dataset(dataset_ref)
dataset.location = 'US'
dataset = client.create_dataset(dataset)  # API request
```
ii) Creating Schema for flowers ML Engine BigQuery
```
SCHEMA = [
    bigquery.SchemaField('KEY', 'INTEGER', mode='REQUIRED'),
    bigquery.SchemaField('PREDICTION', 'INTEGER', mode='REQUIRED'),
    bigquery.SchemaField('SCORE1','FLOAT', mode='REQUIRED'),
    bigquery.SchemaField('SCORE2','FLOAT',mode='REQUIRED'),
    bigquery.SchemaField('SCORE3','FLOAT',mode='REQUIRED'),
    bigquery.SchemaField('SCORE4','FLOAT',mode='REQUIRED'),
    bigquery.SchemaField('SCORE5','FLOAT',mode='REQUIRED'),
    bigquery.SchemaField('SCORE6','FLOAT',mode='REQUIRED'),
    ]
```
iii) Creating a table based on the sample schema
```
schema = [
    bigquery.SchemaField('full_name', 'STRING', mode='REQUIRED'),
    bigquery.SchemaField('age', 'INTEGER', mode='REQUIRED'),
]
table_ref = dataset_ref.table('my_table')
table = bigquery.Table(table_ref, schema=schema)
table = client.create_table(table)  # API request

assert table.table_id == 'my_table'
```
## Insert rows in BigQuery Table
You can load data:

From Cloud Storage
1. From other Google services, such as Google Ad Manager and Google Ads
2. From a readable data source (such as your local machine)
3. By inserting individual records using streaming inserts
4. Using DML statements to perform bulk inserts
5. Using a Google BigQuery IO transform in a Cloud Dataflow pipeline to write data to BigQuery

Note: For the DeepGauge we use streaming inserts.

### Streaming Data into BigQuery
Instead of using a job to load data into BigQuery, you can choose to stream your data into BigQuery one record at a time by using the tabledata().insertAll() method. This approach enables querying data without the delay of running a load job.
```
rows_to_insert = [
    (u'Phred Phlyntstone', 32),
    (u'Wylma Phlyntstone', 29),
]
errors = client.insert_rows(table, rows_to_insert)  # API request

assert errors == []
```
The added table and data can be views at BigQuery WebUI or using BigQuery commands at terminal.
1. WebUI to view the BigQuery datasets and tables for the project
```
https://console.cloud.google.com/bigquery?project=ocideepgauge&authuser=1&p=ocideepgauge&d=flowers_dataset&t=flowers_table&page=table

SELECT SCORE1
FROM `ocideepgauge.flowers_dataset.flowers_table`
LIMIT 1000
```
2. Command line
```
bq ls --format=pretty ocideepgauge:flowers_dataset
```
