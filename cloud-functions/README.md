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
