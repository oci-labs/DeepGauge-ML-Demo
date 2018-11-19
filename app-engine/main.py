# [START gae_python37_render_template]
from flask import Flask, Response, request, json, render_template, current_app
from google.cloud import pubsub_v1, storage
from lib.GCSObjectStreamUpload import GCSObjectStreamUpload
import base64, json, logging, os

app = Flask(__name__)

# Configure the following environment variables via app.yaml
# This is used in the push request handler to veirfy that the request came from
# pubsub and originated from a trusted source.
app.config['PUBSUB_VERIFICATION_TOKEN'] = os.environ['PUBSUB_VERIFICATION_TOKEN']
app.config['PUBSUB_TOPIC'] = os.environ['PUBSUB_TOPIC']
app.config['PROJECT'] = os.environ['GOOGLE_CLOUD_PROJECT']

# Global list to storage messages received by this instance.
MESSAGES = []

@app.route('/')
def root():
    return render_template('dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']

        client = storage.Client()
        bucket = 'ocideepgauge-images'
        with GCSObjectStreamUpload(client=client, bucket_name=bucket, blob_name=file.filename) as s:
            s.write(file.read())

    return "HELLO WORLD"


@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/user_settings')
def user_settings():
    return render_template('user_settings.html')

@app.route('/camera/add')
def add_camera():
    return render_template('add_camera.html')

@app.route('/camera/add/url')
def add_camera_url():
    return 'Add Camera URL'

@app.route('/camera/add/upload')
def add_camera_upload():
    return 'Add Camera Upload'

@app.route('/camera/<int:camera_id>')
def show_camera(camera_id):
    cam = {
        "id": camera_id,
        "img": "https://placehold.it/500x200",
        "acc": 12,
        "type": "Analog Gauge",
        "loc": "St. Louis",
        "notes": "Bacon ipsum dolor amet shank doner jerky short loin filet mignon. Spare ribs short loin turducken jowl."
    }

    return render_template('single_camera.html', camera=cam)

@app.route('/camera/settings/<int:camera_id>')
def show_camera_settings(camera_id):
    cam = {
        "id": camera_id,
        "img": "https://placehold.it/570x200",
        "type": "Analog Gauge",
        "rate": "30",
        "refresh": "60",
        "notes": "Bacon ipsum dolor amet shank doner jerky short loin filet mignon. Spare ribs short loin turducken jowl.",
        "thresholds": [
            { "low": 6 }
        ]
    }
    return render_template('settings_camera.html', camera=cam)

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


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
# [START gae_python37_render_template]
