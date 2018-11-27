import os
from config import db
from models import Setting, Reading, Device, User, Person

# Delete database file if it exists currently
if os.path.exists("deepgauge.db"):
    os.remove("deepgauge.db")

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
