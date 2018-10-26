import os
from flask import Flask, render_template, request
from flask import send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
import subprocess
import Prediction

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(150, 150, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    label,accuracy, annotated_imagepath = Prediction.Predict.predict_online(checkpoint_path='./logs/models/main/',
                                  final_img_width=160,
                                  final_img_height=80,
                                  color_mode="grayscale",filename=full_path)
    print(label,accuracy)
    return(label,accuracy, annotated_imagepath)

# home page
@app.route('/')
def home():
    return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        imagefile='./'+full_name
        #result = api(full_name)
        label,accuracy, annotated_imagepath=api(full_name)
        print(imagefile)
        print(annotated_imagepath)
        print(file.filename)
    return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy,annotated_imagepath=annotated_imagepath)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True

# No cacheing at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response
