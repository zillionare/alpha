from .models import get_current_user, remove_current_user
from .view import layout
from alpha.web import routing


@routing.on("/logout")
def logout():
    """
    logout function.
    """
    if get_current_user():
        remove_current_user()

    return layout


@routing.on("/login")
def login():
    """
    login function.
    """
    return layout
