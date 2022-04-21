from alpha.web import routing

from .models import get_user, remove_session
from .view import render


@routing.dispatch("/logout")
def logout():
    """
    logout function.
    """
    if get_user():
        remove_session()

    return render()


@routing.dispatch("/login")
def login():
    """
    login function.
    """
    return render()
