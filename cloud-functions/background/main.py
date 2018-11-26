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
    print("---- BUCKET ----")
    print(bucket)
    print(data['name'])
    blob = bucket.blob(data['name'])
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

    print(response)

    # Compose request to PUB/SUB
    topic_name = "gauge-prediction"

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project, topic_name)

    # Data must be a bytestring
    predictions = json.dumps(response['predictions'])
    bytestring = predictions.encode('utf-8')

    # Add two attributes, origin and username, to the message
    publisher.publish(topic_path, bytestring, origin='gauge', username='gcp')

    print('Published messages with custom attributes.')

# [END functions_predict_gauge]

# response = {'predictions': [{'probabilities': [0.0, 6.973839958845929e-07, 0.3523220419883728, 2.5842821216315315e-08, 0.006340037100017071, 0.6146539449691772, 7.242937272167183e-07, 9.500018133268284e-11, 8.767632453388075e-11, 4.228942528607513e-19, 5.776533202391594e-26, 3.346123628403849e-18, 2.9990890413472693e-12, 1.1282606230316917e-21, 0.026682548224925995, 3.254650021711214e-11, 8.447079741629853e-17, 1.9302823611722763e-25, 7.598139696144528e-29, 6.168948743934553e-31, 1.3628018158406974e-10, 1.0948157070728376e-15, 2.1688115275496176e-31, 3.548097822184592e-33, 0.0, 0.0, 0.0, 0.0, 2.368963885721671e-36, 3.306142554577295e-26, 7.76042014802086e-15], 'class_ids': [5], 'class_label': ['psi_11'], 'logits': [-101.50701141357422, 0.06559443473815918, 13.198314666748047, -3.2297093868255615, 9.18065357208252, 13.754828453063965, 0.10345592349767685, -8.83561897277832, -8.915844917297363, -28.065641403198242, -43.871883392333984, -25.997203826904297, -12.29118824005127, -33.99208450317383, 10.617778778076172, -9.906826972961426, -22.768600463867188, -42.665435791015625, -50.50553894042969, -55.31908416748047, -8.474783897399902, -20.2066650390625, -56.364437103271484, -60.47737121582031, -75.7141342163086, -90.58768463134766, -147.4109344482422, -96.66671752929688, -67.7890853881836, -44.42990493774414, -18.24821662902832]}]}
#
# print(response)
