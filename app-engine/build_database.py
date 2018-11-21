import os
from config import db
from models import Setting, Reading, Device, User

# Delete database file if it exists currently
if os.path.exists("deepgauge.db"):
    os.remove("deepgauge.db")

# Create the database
db.create_all()
