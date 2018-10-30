import base64, json
from google.cloud import storage

# [START functions_predict_gauge]
def predict_gauge(data, context):
    """Background Cloud Function to be triggered by Cloud Storage.
       This generic function logs relevant data when a file is changed.

    Args:
        data (dict): The Cloud Functions event payload.
        context (google.cloud.functions.Context): Metadata of triggering event.
    Returns:
        None; the output is written to Stackdriver Logging
    """

    # Get the file from the storage bucket
    client = storage.Client()
    bucket_id = 'ocideepgauge-images'
    bucket = client.get_bucket(bucket_id)
    blob = bucket.blob(data['name'])
    img = base64.b64encode(blob.download_as_string())
    json_dumps = json.dumps({"key":"0", "image_bytes": {"b64": img}})

    print("NEWER VERSION")
    # print(json_dumps)
    # print("IMG".format(type(blob)))

    # print( 'Bucket: {}'.format(type(bucket)) )
    # print( 'JSON DUMPS: {}'.format(json_dumps) )

    print('Event ID: {}'.format(context.event_id))
    print('Event type: {}'.format(context.event_type))
    print('Bucket: {}'.format(data['bucket']))
    print('File: {}'.format(data['name']))
    print('Metageneration: {}'.format(data['metageneration']))
    print('Created: {}'.format(data['timeCreated']))
    print('Updated: {}'.format(data['updated']))

# [END functions_predict_gauge]
