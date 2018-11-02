import base64, json
from google.cloud import storage
from googleapiclient import discovery
from google.cloud import pubsub_v1


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

    instance = {"key":"0", "image_bytes": {"b64": img.decode("utf8")}}

    #
    # Compose request to ML Engine
    #
    project = 'ocideepgauge'
    model = 'flowers'
    service = discovery.build('ml', 'v1', cache_discovery=False)
    name = 'projects/{}/models/{}'.format(project, model)

    # Create a request to call projects.models.predict.
    response = service.projects().predict(
        name=name,
        body={'instances': [instance]}
    ).execute()

    #
    # Compose request to PUB/SUB
    #
    topic_name = "flower-prediction"

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project, topic_name)

    # Data must be a bytestring
    predictions = json.dumps(response['predictions'])
    bytestring = predictions.encode('utf-8')

    # Add two attributes, origin and username, to the message
    publisher.publish(topic_path, bytestring, origin='flower-sample', username='gcp')

    print('Published messages with custom attributes.')

    # daisy - 0, dandelion - 1, roses - 2, sunflowers - 3, tulips - 4
    # print(response['predictions'])
    # Print General Information
    # print('Event ID: {}'.format(context.event_id))
    # print('Event type: {}'.format(context.event_type))
    # print('Bucket: {}'.format(data['bucket']))
    # print('File: {}'.format(data['name']))
    # print('Metageneration: {}'.format(data['metageneration']))
    # print('Created: {}'.format(data['timeCreated']))
    # print('Updated: {}'.format(data['updated']))

# [END functions_predict_gauge]
