"""
This is the people module and supports all the REST actions for the
people data
"""

from flask import make_response, abort
from config import db
from models import Default, DefaultSchema


def read_all():
    """
    This function responds to a request for /api/default
    with the complete lists of users defaults

    :return:        json string of list of default settings
    """
    # Create the list of people from our data
    users = Default.query.order_by(Default.id_user).all()

    # Serialize the data for the response
    default_schema = DefaultSchema(many=True)
    data = default_schema.dump(users).data
    return data


def read_one(id_user):
    """
    This function responds to a request for /api/default/{id_user}
    with one matching user from default settings

    :param id_user:   Id of user to find
    :return:            user matching id
    """
    # Get the person requested
    defaults = Default.query.filter(Default.id_user == id_user).one_or_none()

    # Did we find a person?
    if defaults is not None:

        # Serialize the data for the response
        default_schema = DefaultSchema()
        data = default_schema.dump(defaults).data
        return data

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Defaults not found for Id: {id_user}".format(id_user=id_user),
        )


def create(default):
    """
    This function creates a new entry in the default structure
    based on the passed in user id data

    :param default:  person to create in people structure
    :param default.id_user: user id to create the default structure
    :param default.type: default type to display for the user
    :param default.frame_rate: default frame rate of the device
    :param default.refresh_rate: refresh rate of the device
    :param default.updated: timestamp of the last update
    :return:        201 on success, 406 on default exists
    """

    id_user = default.get("id_user")
    type = default.get("type")

    existing_default = (
        Default.query.filter(Default.id_user == id_user)
        .filter(Default.type == type)
        .one_or_none()
    )

    # Can we insert this person?
    if existing_default is None:

        # Create a person instance using the schema and the passed in person
        schema = DefaultSchema()
        new_default = schema.load(default, session=db.session).data

        # Add the person to the database
        db.session.add(new_default)
        db.session.commit()

        # Serialize and return the newly created person in the response
        data = schema.dump(new_default).data

        return data, 201

    # Otherwise, nope, person exists already
    else:
        abort(
            409,
            "Default {id_user} exists already".format(
                id_user=id_user
            ),
        )


def update(id_user, default):
    """
    This function updates an existing setting in the default structure

    :param id_user:   id of the user to update in the default structure
    :param default:   setting to update
    :return:          updated default structure
    """
    # Get the person requested from the db into session
    update_default = Default.query.filter(
        Default.id_user == id_user
    ).one_or_none()

    # Did we find a user?
    if update_default is not None:

        # turn the passed in person into a db object
        schema = DefaultSchema()
        update = schema.load(default, session=db.session).data

        # Set the id to the person we want to update
        update.id = update_default.id_user

        # merge the new object into the old and commit it to the db
        db.session.merge(update)
        db.session.commit()

        # return updated person in the response
        data = schema.dump(update_default).data

        return data, 200

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Default not found for Id: {id_user}".format(id_user=id_user),
        )


def delete(id_user):
    """
    This function deletes a user from the default structure

    :param id_user:     Id of the user to delete
    :return:            200 on successful delete, 404 if not found
    """
    # Get the person requested
    defaults = Default.query.filter(Default.id_user == id_user).one_or_none()

    # Did we find a person?
    if defaults is not None:
        db.session.delete(defaults)
        db.session.commit()
        return make_response(
            "Default {id_user} deleted".format(id_user=id_user), 200
        )

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Default not found for Id: {id_user}".format(id_user=id_user),
        )
