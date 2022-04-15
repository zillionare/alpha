from .view import layout
from alpha.web import routing


@routing.on("/")
def render_home_page():
    return layout
