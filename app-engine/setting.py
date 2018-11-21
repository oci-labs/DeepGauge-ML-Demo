"""
This is the people module and supports all the REST actions for the
people data
"""

from flask import make_response, abort
from config import db
from models import Setting, SettingSchema


def read_all():
    """
    This function responds to a request for /api/setting
    with the complete lists of users settings

    :return:        json string of list of default settings
    """
    # Create the list of people from our data
    users = Setting.query.order_by(Setting.id_user).all()

    # Serialize the data for the response
    settings_schema = SettingSchema(many=True)
    data = settings_schema.dump(users).data
    return data


def read_one(id_user):
    """
    This function responds to a request for /api/setting/{id_user}
    with one matching user from settings

    :param id_user:   Id of user to find
    :return:            user matching id
    """
    # Get the person requested
    settings = Setting.query.filter(Setting.id_user == id_user).one_or_none()

    # Did we find a person?
    if settings is not None:

        # Serialize the data for the response
        setting_schema = SettingSchema()
        data = setting_schema.dump(settings).data
        return data

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Settings not found for Id: {id_user}".format(id_user=id_user),
        )


def create(setting):
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

    id_user = setting.get("id_user")
    type = setting.get("type")

    existing_setting = (
        Setting.query.filter(Setting.id_user == id_user)
        .one_or_none()
    )

    # Can we insert this person?
    if existing_setting is None:

        # Create a person instance using the schema and the passed in person
        schema = SettingSchema()
        new_setting = schema.load(setting, session=db.session).data

        # Add the person to the database
        db.session.add(new_setting)
        db.session.commit()
    #
    #     # Serialize and return the newly created person in the response
    #     data = schema.dump(new_setting).data
    #
    #     return data, 201
    #
    # # Otherwise, nope, person exists already
    # else:
    #     abort(
    #         409,
    #         "Setting {id_user} exists already".format(
    #             id_user=id_user
    #         ),
    #     )


def update(id_user, default):
    """
    This function updates an existing setting in the default structure

    :param id_user:   id of the user to update in the default structure
    :param setting:   setting to update
    :return:          updated default structure
    """
    # Get the person requested from the db into session
    update_setting = Setting.query.filter(
        Setting.id_user == id_user
    ).one_or_none()

    # Did we find a user?
    if update_setting is not None:

        # turn the passed in person into a db object
        schema = SettingSchema()
        update = schema.load(setting, session=db.session).data

        # Set the id to the person we want to update
        update.id = update_setting.id_user

        # merge the new object into the old and commit it to the db
        db.session.merge(update)
        db.session.commit()

        # return updated person in the response
        data = schema.dump(update_setting).data

        return data, 200

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Setting not found for Id: {id_user}".format(id_user=id_user),
        )


def delete(id_user):
    """
    This function deletes a user from the default structure

    :param id_user:     Id of the user to delete
    :return:            200 on successful delete, 404 if not found
    """
    # Get the person requested
    setting = Setting.query.filter(Setting.id_user == id_user).one_or_none()

    # Did we find a person?
    if setting is not None:
        db.session.delete(setting)
        db.session.commit()
        return make_response(
            "Setting {id_user} deleted".format(id_user=id_user), 200
        )

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "Setting not found for Id: {id_user}".format(id_user=id_user),
        )
