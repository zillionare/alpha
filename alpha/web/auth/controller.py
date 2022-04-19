from .models import get_current_user, remove_current_user
from .view import render
from alpha.web import routing


@routing.on("/logout")
def logout(sid: str):
    """
    logout function.
    """
    if get_current_user():
        remove_current_user()

    return render()


@routing.on("/login")
def login(sid: str):
    """
    login function.
    """
    return render()
