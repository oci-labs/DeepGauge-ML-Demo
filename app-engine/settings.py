"""
This is the people module and supports all the REST actions for the
settings data
"""

from flask import make_response, abort
from config import db
from models import Settings, SettingSchema

def create(setting):
    """
    This function creates a new setting in the settings structure
    based on the passed in setting data

    :param setting:  setting to create in settings structure
    :return:        201 on success, 406 on setting exists
    """
    frame = setting.get("frame")
    refresh = setting.get("refresh")
    type = setting.get("type")

    # Create a person instance using the schema and the passed in person
    schema = SettingSchema()
    new_setting = schema.load(setting, session=db.session).data

    # Add the person to the database
    db.session.add(new_setting)
    db.session.commit()

    # Serialize and return the newly created person in the response
    data = schema.dump(new_setting).data

    return data, 201
