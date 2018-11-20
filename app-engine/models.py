from datetime import datetime
from config import db, ma

class Settings(db.Model):
    __tablename__ = "person"
    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(32))
    frame = db.Column(db.Integer)
    refresh = db.Column(db.Integer)


class SettingsSchema(ma.ModelSchema):
    class Meta:
        model = Settings
        sqla_session = db.session
