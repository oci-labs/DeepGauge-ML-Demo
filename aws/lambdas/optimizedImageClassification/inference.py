#
# Copyright 2010-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Greengrass lambda function to perform Image Classification with example model
# Resnet-50 that was compiled by DLC.
#
#
import logging
import os
import json

from dlr import DLRModel
from PIL import Image
import numpy as np

import greengrasssdk
import camera
import utils

import boto3

kinesis = boto3.client('kinesis')

# Create MQTT client
mqtt_client = greengrasssdk.client('iot-data')

# Initialize logger
customer_logger = logging.getLogger(__name__)

# Initialize example Resnet-50 model
model_resource_path = os.environ.get('MODEL_PATH', '/ml_model')
input_shape = {'data': [1, 3, 224, 224]}
output_shape = [1, 1000]
dlr_model = DLRModel(model_resource_path, input_shape, output_shape, 'cpu')

# Read synset file
#synset_path = os.path.join(model_resource_path, 'synset.txt')
#with open(synset_path, 'r') as f:
#    synset = eval(f.read())


def predict(image_data):
    r"""
    Predict image with DLR. The result will be published
    to MQTT topic 'deepgauge_model/predictions'.

    :param image: numpy array of the Image inference with.
    """
    flattened_data = image_data.astype(np.float32).flatten()

    prediction_scores = dlr_model.run({'data' : flattened_data}).squeeze()
    max_score_id = np.argmax(prediction_scores)
    max_score = np.max(prediction_scores)

    result = json.dumps({"id": max_score_id, "score": max_score})

    # Prepare result
    #predicted_class = synset[max_score_id]
    #result = 'Inference result: probability {}.'.format(max_score)

    # Send result
    send_mqtt_message(result)
    kinesis.put_record(StreamName="DeepGauge",
                       Data=result,
                       PartitionKey="singleshardkey")


def predict_from_cam():
    r"""
    Predict with the photo taken from your pi camera.
    """
    #send_mqtt_message("Taking a photo...")
    my_camera = camera.Camera()
    image = Image.open(my_camera.capture_image())
    image_data = utils.transform_image(image)

    #send_mqtt_message("Start predicting...")
    predict(image_data)


def predict_from_image(filename):
    image = Image.open(filename)
    image_data = utils.transform_image(image)

    #send_mqtt_message("Start predicting...")
    predict(image_data)
    

def send_mqtt_message(message):
    r"""
    Publish message to the MQTT topic:
    'deepgauge_model/predictions'.

    :param message: message to publish
    """
    mqtt_client.publish(topic='deepgauge_model/predictions',
                        payload=message)


# The lambda to be invoked in Greengrass
def handler(event, context):
    filename = event.get('filename')
    try:
        if (filename == None):
            predict_from_cam()
        else:
            predict_from_image(filename)

    except Exception as e:
        customer_logger.exception(e)
        send_mqtt_message(
            'Exception occurred during prediction. Please check logs for troubleshooting: /greengrass/ggc/var/log.')
