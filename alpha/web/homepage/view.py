from dash import Input, Output, callback, dcc, html

layout = html.Div(
    [
        dcc.Location(id="homepage", refresh=False),
        html.H1("Welcome to Alhpa Homepage!"),
        dcc.Link("Logout", href="/logout"),
    ]
)
