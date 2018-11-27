# [START gae_python37_render_template]
from flask import Flask, Response, request, json, render_template, current_app, redirect
from google.cloud import pubsub_v1, storage
from lib.GCSObjectStreamUpload import GCSObjectStreamUpload
import base64, json, logging, os
from config import db, app, connex_app
from models import *

# Read the swagger.yml file to configure the endpoints
connex_app.add_api("swagger.yml")

def make_database():
    # Delete database file if it exists currently
    # if os.path.exists("deepgauge.db"):
    #     os.remove("deepgauge.db")

    # Create the database
    db.create_all()

    # Data to initialize database with
    DEVICES = [
        {
            "id_user":1,
            "name":"Device One",
            "image":"https://placehold.it/282x282/",
            "bucket":"ocideepgauge",
            "type":"Gauge",
            "location":"St. Louis",
            "frame_rate":5,
            "refresh_rate":60,
            "notes":"General notes and information about Camera One",
            "high_threshold":10,
            "low_threshold":5
        }
    ]

    # iterate over the PEOPLE structure and populate the database
    for device in DEVICES:
        d = Device(
            id_user=device.get("id_user"),
            name=device.get("name"),
            image=device.get("image"),
            bucket=device.get("bucket"),
            type=device.get("type"),
            location=device.get("location"),
            frame_rate=device.get("frame_rate"),
            refresh_rate=device.get("refresh_rate"),
            notes=device.get("notes"),
            high_threshold=device.get("high_threshold"),
            low_threshold=device.get("low_threshold")
        )
        db.session.add(d)

    u = User(
        user_name       = "Technician",
        display_name    = "Technician Name",
        company         = "Technicians Company",
        thumbnail       = "https://jobs.centurylink.com/sites/century-link/images/sp-technician-img.jpg"
    )
    db.session.add(u)

    r = Reading(
        id_device   = 1,
        prediction  = "psi 8",
        accuracy    = "89%",
        body        = "[{}]"
    )
    db.session.add(r)

    db.session.commit()
    return True

# Flask lets you create a test request to initialize an app.
with app.test_request_context():
     make_database()

@app.route('/')
def root():
    query = Device.query.order_by(Device.id_user).all()

    # Serialize the data for the response
    schema = DeviceSchema(many=True)
    data = schema.dump(query).data

    return render_template('dashboard.html', devices=data)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        client = storage.Client()
        bucket = 'ocideepgauge-images' #TODO: Make this a global variable

        # Create a new Device entry
        schema = DeviceSchema()
        device = Device(
            id_user         = 1, #TODO request this from the Flask auth session - not implemented
            name            = "",
            image           = 'https://storage.googleapis.com/{0}/{1}'.format(bucket, file.filename),
            bucket          = "gs://ocideepgauge-images",
            type            = "gauge",
            location        = "",  #TODO detect or update value from geo service
            frame_rate      = 15, #TODO change to defaults set in the database
            refresh_rate    = 30, #TODO change to defaults set in the database
            notes           = "",
            high_threshold  = 0,
            low_threshold   = 0
        )

        # Add the device to the database
        db.session.add(device)
        db.session.commit()

        # Serialize and return the newly created person in the response
        data = schema.dump(device).data

        # Upload the image to Google Storage bucket
        with GCSObjectStreamUpload(client=client, bucket_name=bucket, blob_name=file.filename) as s:
            s.write(file.read())

        my_bucket = client.get_bucket(bucket)
        blob = my_bucket.blob(file.filename)
        blob.metadata = {'device_id':data['id'],'type':'gauge'}
        blob.patch()

        # Redirect to the device page.
        return redirect("/device/setting/{}".format(data['id']), code=302)


@app.route('/setting')
def setting():
    ## TODO Add query to local database to get defaults
    return render_template('setting.html')

@app.route('/user')
def user():
    user = User.query.filter(User.id == 1).one_or_none()

    # Serialize the data for the response
    schema = UserSchema()
    data = schema.dump(user).data

    return render_template('user.html', user=data)

@app.route('/device/new')
def new_device():
    return render_template('new_device.html')

@app.route('/device/<int:device_id>')
def one_device(device_id):
    query = Device.query.filter(Device.id == device_id).one_or_none()
    if query is not None:

        # Serialize the data for the response
        schema = DeviceSchema()
        data = schema.dump(query).data

    # Otherwise, nope, didn't find that person
    else:
        data = []

    query_reading = Reading.query.filter(Reading.id_device == device_id).one_or_none()
    if query_reading is not None:

        # Serialize the data for the response
        schema = ReadingSchema()
        reading = schema.dump(query_reading).data

    # Otherwise, nope, didn't find that person
    else:
        reading = []

    print(reading)
    return render_template('one_device.html', device=data, reading=reading)

@app.route('/device/setting/<int:device_id>')
def show_device_setting(device_id):
    query = Device.query.filter(Device.id == device_id).one_or_none()

    # Did we find a person?
    if query is not None:

        # Serialize the data for the response
        schema = DeviceSchema()
        data = schema.dump(query).data

    # Otherwise, nope, didn't find that person
    else:
        data = []
    print(data)
    return render_template('setting_device.html', device=data)

# [START push]
@app.route('/pubsub/push', methods=['POST'])
def pubsub_push():
    if (request.args.get('token', '') !=
            current_app.config['PUBSUB_VERIFICATION_TOKEN']):
        return 'Invalid request', 400

    envelope = json.loads(request.get_data().decode('utf-8'))
    payload = base64.b64decode(envelope['message']['data'])

    payload_json = payload.decode('utf8').replace("'", '"')
    payload_data = json.loads(payload_json)
    for d in payload_data:
        for cl in d['class_label']:
            prediction = cl

    schema = ReadingSchema()
    reading = Reading(
        id_device   = envelope['message']['attributes']['device'],
        prediction  = prediction,
        accuracy    = "",
        body        = payload_json
    )
    # Add to the database
    db.session.add(reading)
    db.session.commit()

    # Serialize and return the newly created person in the response
    data = schema.dump(reading).data

    # Returning any 2xx status indicates successful receipt of the message.
    return 'OK', 200
# [END push]


# @app.errorhandler(500)
# def server_error(e):
#     logging.exception('An error occurred during a request.')
#     return """
#     An internal error occurred: <pre>{}</pre>
#     See logs for full stacktrace.
#     """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
# [START gae_python37_render_template]
