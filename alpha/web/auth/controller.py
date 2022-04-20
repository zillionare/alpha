from .models import get_user, remove_session
from .view import render
from alpha.web import routing


@routing.on("/logout")
def logout():
    """
    logout function.
    """
    if get_user():
        remove_session()

    return render()


@routing.on("/login")
def login():
    """
    login function.
    """
    return render()
