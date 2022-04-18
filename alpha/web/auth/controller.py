from .models import get_current_user, remove_current_user
from .view import render
from alpha.web import routing


@routing.on("/logout")
def logout():
    """
    logout function.
    """
    if get_current_user():
        remove_current_user()

    return render()


@routing.on("/login")
def login():
    """
    login function.
    """
    return render()
