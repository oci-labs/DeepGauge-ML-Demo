#
# Copyright 2010-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# This class is a camera that uses picamera to take a photo and DLC compiled
# Resnet-50 model to perform image classification, identifying the objects
# shown in the photo.
#

import StringIO
import picamera
import time

class Camera(object):
    r"""
    Camera that captures an image for performing inference
    with DLC compiled model.
    """

    def capture_image(self):
        r"""
        Capture image with PiCamera.
        """
        camera = picamera.PiCamera()
        imageData = StringIO.StringIO()

        try:
            camera.resolution = (224, 224)
            print("Taking a photo from your camera...")
            camera.start_preview()
            time.sleep(2)
            camera.capture(imageData, format = "jpeg", resize = (224, 224))
            camera.stop_preview()

            imageData.seek(0)
            return imageData
        finally:
            camera.close()

        raise RuntimeError("There is problem to use your camera.")
