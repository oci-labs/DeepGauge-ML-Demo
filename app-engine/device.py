"""
This is the people module and supports all the REST actions for the
people data
"""

from flask import make_response, abort
from config import db
from models import Device, DeviceSchema
from datetime import datetime

def read_all():
    """
    This function responds to a request for /api/device
    with the complete lists of users

    :return:        json string of list of devices
    """
    # Create the list of people from our data
    query = Device.query.order_by(Device.id_user).all()

    # Serialize the data for the response
    schema = DeviceSchema(many=True)
    data = schema.dump(query).data
    return data


def read_one(id_device):
    """
    This function responds to a request for /api/device/{id_user}
    with one matching user from settings

    :param id_user:   Id of user to find
    :return:            user matching id
    """
    # Get the person requested
    query = Device.query.filter(Device.id == id_device).one_or_none()

    # Did we find a person?
    if query is not None:

        # Serialize the data for the response
        schema = DeviceSchema()
        data = schema.dump(query).data
        return data

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Device not found for Id: {id_user}".format(id_user=id_user),
        )


def create(device):
    """
    This function creates a new entry in the device structure
    based on the passed in user id data

    :param user_name:  user name to create in user structure
    :param display_name:  display name for the user
    :param company:  the user company
    :param thumbnail:  string url to the thumbnail image

    :return:        201 on success, 406 on default exists
    """
    # Create a person instance using the schema and the passed in person
    schema = DeviceSchema()
    devices = schema.load(device, session=db.session).data

    # Add the person to the database
    db.session.add(devices)
    db.session.commit()

    # Serialize and return the newly created person in the response
    data = schema.dump(devices).data

    return data, 201



def update(id_device, device):
    """
    This function updates an existing user in the structure

    :param id_device:   id of the device to update in the default structure
    :param device:   device to update
    :return:       updated device structure
    """
    # Get the person requested from the db into session
    update_device = Device.query.filter(Device.id == id_device).one_or_none()

    # Did we find a device?
    if update_device is not None:
        update_device.updated = datetime.utcnow()
        update_device.bucket = device['bucket']
        update_device.frame_rate = device['frame_rate']
        update_device.high_threshold = device['high_threshold']
        update_device.id_user = device['id_user']
        update_device.image = device['image']
        update_device.location = device['location']
        update_device.low_threshold = device['low_threshold']
        update_device.name = device['name']
        update_device.notes = device['notes']
        update_device.prediction = device['prediction']
        update_device.refresh_rate = device['refresh_rate']
        update_device.type = device['type']

        db.session.commit()

        schema = DeviceSchema()
        data = schema.dump(update_device).data

        return data, 200

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Device not found for Id: {id_device}".format(id_device=id_device),
        )


def delete(id_device):
    """
    This function deletes a user from the default structure

    :param user_name:     Id of the user to delete
    :return:            200 on successful delete, 404 if not found
    """
    # Get the person requested
    delete = Device.query.filter(Device.id == id_device).one_or_none()

    # Did we find a person?
    if delete is not None:
        db.session.delete(delete)
        db.session.commit()
        return make_response(
            "Device {id_device} deleted".format(id_device=id_device), 200
        )

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Device not found for Id: {id_device}".format(id_device=id_device),
        )
