# [START gae_python37_render_template]
from flask import Flask, Response, request, json, render_template, current_app, redirect
from google.cloud import pubsub_v1, storage
from lib.GCSObjectStreamUpload import GCSObjectStreamUpload
import base64, json, logging, os
from config import db, app, connex_app
from models import Setting, Reading, User, Person, Device, DeviceSchema

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
            "type":"Camera",
            "location":"St. Louis",
            "frame_rate":5,
            "refresh_rate":60,
            "notes":"General notes and information about Camera One",
            "high_threshold":10,
            "low_threshold":5
        },
        {
            "id_user":1,
            "name":"Device Two",
            "image":"https://placehold.it/282x282/",
            "bucket":"ocideepgauge",
            "type":"Camera",
            "location":"St. Louis",
            "frame_rate":10,
            "refresh_rate":120,
            "notes":"General notes and information about Camera One",
            "high_threshold":20,
            "low_threshold":10
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

    db.session.commit()
    return True

@app.route('/build-my-database')
def database():
    make_database()
    return 'OK', 200

@app.route('/')
def root():
    query = Device.query.order_by(Device.id_user).all()

    # Serialize the data for the response
    schema = DeviceSchema(many=True)
    data = schema.dump(query).data
    print(data)
    return render_template('dashboard.html', devices=data)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']

        client = storage.Client()
        ## TODO: Make this a global variable
        bucket = 'ocideepgauge-images'
        ##
        with GCSObjectStreamUpload(client=client, bucket_name=bucket, blob_name=file.filename) as s:
            s.write(file.read())
        ## Create a new Device
        ## Redirect to the device page.

    return redirect("/", code=302)


@app.route('/setting')
def setting():
    ## TODO Add query to local database to get defaults
    return render_template('setting.html')

@app.route('/user')
def user():
    return render_template('user.html')

@app.route('/device/new')
def new_device():
    return render_template('new_device.html')

@app.route('/device/<int:device_id>')
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

@app.route('/device/setting/<int:device_id>')
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
@app.route('/pubsub/push', methods=['POST'])
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
