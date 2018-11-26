# [START gae_python37_render_template]
from flask import Flask, Response, request, json, render_template, current_app, redirect
from google.cloud import pubsub_v1, storage
from lib.GCSObjectStreamUpload import GCSObjectStreamUpload
import base64, json, logging, os

from models import Device, DeviceSchema

# local modules
import config

# Get the application instance
connex_app = config.connex_app

# Read the swagger.yml file to configure the endpoints
connex_app.add_api("swagger.yml")


@connex_app.route('/')
def root():
    query = Device.query.order_by(Device.id_user).all()

    # Serialize the data for the response
    schema = DeviceSchema(many=True)
    data = schema.dump(query).data
    print(data)
    return render_template('dashboard.html', devices=data)

@connex_app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']

        client = storage.Client()
        ## TODO: Make this a global variable
        bucket = 'ocideepgauge-images'
        ##
        with GCSObjectStreamUpload(client=client, bucket_name=bucket, blob_name=file.filename) as s:
            s.write(file.read())

    return redirect("/", code=302)


@connex_app.route('/setting')
def setting():
    ## TODO Add query to local database to get defaults
    return render_template('setting.html')

@connex_app.route('/user')
def user():
    return render_template('user.html')

@connex_app.route('/device/new')
def new_device():
    return render_template('new_device.html')

@connex_app.route('/device/<int:device_id>')
def one_device(device_id):
    query = Device.query.filter(Device.id == device_id).one_or_none()

    # Did we find a person?
    if query is not None:

        # Serialize the data for the response
        schema = DeviceSchema()
        data = schema.dump(query).data

    # Otherwise, nope, didn't find that person
    else:
        data = []

    return render_template('one_device.html', device=data)

@connex_app.route('/device/setting/<int:device_id>')
def show_device_setting(device_id):
    obj = {
        "id": device_id,
        "img": "https://placehold.it/570x200",
        "type": "Analog Gauge",
        "rate": "30",
        "refresh": "60",
        "notes": "Bacon ipsum dolor amet shank doner jerky short loin filet mignon. Spare ribs short loin turducken jowl.",
        "thresholds": [
            { "low": 6, "high": 18 }
        ]
    }
    return render_template('setting_device.html', device=obj)

# [START push]
@connex_app.route('/pubsub/push', methods=['POST'])
def pubsub_push():
    if (request.args.get('token', '') !=
            current_app.config['PUBSUB_VERIFICATION_TOKEN']):
        return 'Invalid request', 400

    envelope = json.loads(request.get_data().decode('utf-8'))
    payload = base64.b64decode(envelope['message']['data'])

    MESSAGES.append(payload)

    # Returning any 2xx status indicates successful receipt of the message.
    return 'OK', 200
# [END push]


# @connex_app.errorhandler(500)
# def server_error(e):
#     logging.exception('An error occurred during a request.')
#     return """
#     An internal error occurred: <pre>{}</pre>
#     See logs for full stacktrace.
#     """.format(e), 500


if __name__ == '__main__':
    connex_app.run(host='127.0.0.1', port=8080, debug=True)
# [START gae_python37_render_template]
