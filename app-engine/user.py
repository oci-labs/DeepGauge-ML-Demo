"""
This is the people module and supports all the REST actions for the
people data
"""

from flask import make_response, abort
from config import db
from models import User, UserSchema


def read_all():
    """
    This function responds to a request for /api/user
    with the complete lists of users

    :return:        json string of list of users
    """
    # Create the list of people from our data
    users = User.query.order_by(User.user_name).all()

    # Serialize the data for the response
    schema = UserSchema(many=True)
    data = schema.dump(users).data
    return data


def read_one(user_name):
    """
    This function responds to a request for /api/user/{user_name}
    with one matching user from settings

    :param id_user:   Id of user to find
    :return:            user matching id
    """
    # Get the person requested
    user = User.query.filter(User.user_name == user_name).one_or_none()

    # Did we find a person?
    if user is not None:

        # Serialize the data for the response
        schema = UserSchema()
        data = schema.dump(user).data
        return data

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "User not found for Id: {user_name}".format(user_name=user_name),
        )


def create(user):
    """
    This function creates a new entry in the default structure
    based on the passed in user id data

    :param user_name:  user name to create in user structure
    :param display_name:  display name for the user
    :param company:  the user company
    :param thumbnail:  string url to the thumbnail image

    :return:        201 on success, 406 on default exists
    """


    user_name = user.get("user_name")

    existing = (
        User.query.filter(User.user_name == user_name)
        .one_or_none()
    )

    # Can we insert this person?
    if existing is None:

        # Create a person instance using the schema and the passed in person
        schema = UserSchema()
        users = schema.load(user, session=db.session).data

        # Add the person to the database
        db.session.add(users)
        db.session.commit()

        # Serialize and return the newly created person in the response
        data = schema.dump(users).data

        return data, 201

    # Otherwise, nope, person exists already
    else:
        abort(
            409,
            "User {user_name} exists already".format(
                user_name=user_name
            ),
        )


def update(user_name, user):
    """
    This function updates an existing user in the structure

    :param user_name:   id of the user to update in the default structure
    :param user:   user to update
    :return:       updated default structure
    """
    # Get the person requested from the db into session
    update = User.query.filter(
        User.user_name == user_name
    ).one_or_none()

    # Did we find a user?
    if update is not None:

        # turn the passed in person into a db object
        schema = UserSchema()
        updates = schema.load(user, session=db.session).data

        # Set the id to the person we want to update
        updates.user_name = user_name

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
            "User not found for Id: {user_name}".format(user_name=user_name),
        )


def delete(user_name):
    """
    This function deletes a user from the default structure

    :param user_name:     Id of the user to delete
    :return:            200 on successful delete, 404 if not found
    """
    # Get the person requested
    user = User.query.filter(User.user_name == user_name).one_or_none()

    # Did we find a person?
    if user is not None:
        db.session.delete(user)
        db.session.commit()
        return make_response(
            "User {user_name} deleted".format(user_name=user_name), 200
        )

    # Otherwise, nope, didn't find that person
    else:
        abort(
            404,
            "User not found for Id: {user_name}".format(user_name=user_name),
        )
