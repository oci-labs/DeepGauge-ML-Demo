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
gcloud functions deploy predict_gauge --source=. --runtime python37 --trigger-resource ocideepgauge-images --trigger-event google.storage.object.finalize
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
