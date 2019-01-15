import base64, json, datetime

# Imports the Google Cloud client library
# from google.cloud import bigquery
from google.cloud import pubsub_v1
from google.cloud import storage
from googleapiclient import discovery


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
    my_blob = bucket.get_blob(data['name'])
    img = base64.b64encode(blob.download_as_string())

    instance = {"bytes": {"b64": img.decode("utf8")}}

    # Compose request to ML Engine
    project = 'ocideepgauge'
    model = 'dg'
    service = discovery.build('ml', 'v1', cache_discovery=False)
    name = 'projects/{}/models/{}'.format(project, model)

    # Create a request to call projects.models.predict.
    response = service.projects().predict(
        name=name,
        body={'instances': [instance]}
    ).execute()

    # Compose request to PUB/SUB
    topic_name = "gauge-prediction"

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project, topic_name)

    # return the image of the device
    thumbnail = 'https://storage.googleapis.com/{0}/{1}'.format(bucket_id, data['name'])

    # Data must be a bytestring
    predictions = json.dumps(response['predictions'])
    bytestring = predictions.encode('utf-8')
    device_id = my_blob.metadata['device_id']


    # Add two attributes, origin and username, to the message
    publisher.publish(topic_path,bytestring,image=thumbnail,device=device_id)

    return response

# [END functions_predict_gauge]
