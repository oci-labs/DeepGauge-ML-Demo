# Reading
#   id
#   id_device
#   prediction
#   accuracy
#   body
#   timestamp

"""
This is the people module and supports all the REST actions for the
people data
"""

from flask import make_response, abort
from config import db
from models import Reading, ReadingSchema


def read_all():
    """
    This function responds to a request for /api/user
    with the complete lists of users

    :return:        json string of list of users
    """
    # Create the list of people from our data
    users = Reading.query.order_by(Reading.id_device).all()

    # Serialize the data for the response
    schema = ReadingSchema(many=True)
    data = schema.dump(users).data
    return data


def read_many(id_device):
    """
    This function responds to a request for /api/reading/{id_device}
    with matching readings from device id

    :param id_device:   Id of device to find
    :return:            readings matching id
    """
    # Get the person requested
    read = Reading.query.filter(Reading.id_device == id_device).all()

    # Did we find a person?
    if read is not None:

        # Serialize the data for the response
        schema = ReadingSchema(many=True)
        data = schema.dump(read).data
        return data, 201

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Readings not found for Id: {id_device}".format(id_device=id_device),
        )


def create(reading):
    """
    This function creates a new entry in the default structure
    based on the passed in user id data

    :param id_device:  individual device identification
    :param prediction:  the predicted label
    :param accuracy:  the accuracy of the predictions
    :param body:  the reccommendation response body

    :return:        201 on success
    """
    # Create a person instance using the schema and the passed in person
    schema = ReadingSchema()
    readings = schema.load(reading, session=db.session).data
    readings.id_device = reading.get('id_device')
    # Add to the database
    db.session.add(readings)
    db.session.commit()

    # Serialize and return the newly created person in the response
    data = schema.dump(reading).data

    return data, 201


def update(reading_id, reading):
    """
    This function updates an existing reading in the structure

    :param id:   id of the reading to update in the default structure
    :param user:   user to update
    :return:       updated default structure
    """
    # Get the person requested from the db into session
    update = Reading.query.filter(
        Reading.id == reading_id
    ).one_or_none()

    # Did we find a user?
    if update is not None:

        # turn the passed in person into a db object
        schema = ReadingSchema()
        updates = schema.load(reading, session=db.session).data

        # merge the new object into the old and commit it to the db
        db.session.merge(updates)
        db.session.commit()

        # return updated person in the response
        data = schema.dump(updates).data

        return data, 200

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Reading not found for Id: {reading_id}".format(reading_id=reading_id),
        )


def delete(reading_id):
    """
    This function deletes a reading from the default structure

    :param reading_id:     Id of the reading to delete
    :return:            200 on successful delete, 404 if not found
    """
    # Get the row requested
    row = Reading.query.filter(Reading.id == reading_id).one_or_none()

    # Did we find a person?
    if row is not None:
        db.session.delete(row)
        db.session.commit()
        return make_response(
            "Id {reading_id} deleted".format(reading_id=reading_id), 200
        )

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Reading not found for Id: {reading_id}".format(reading_id=reading_id)
        )
