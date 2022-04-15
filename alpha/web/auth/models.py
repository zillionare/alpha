import flask
import uuid

user_pwd = {
    "dash": "123",
    "dash1": "1234",
}
user_names = {
    "dash": "User1, welcome to the crypto indicators dashboard",
    "dash1": "User1, welcome to the crypto indicators dashboard",
}


def users_info():
    return user_pwd, user_names


user_sessions = {}


def login_user(response, username, password):
    global user_sessions

    cookie = flask.request.cookies.get("zillionare-auth-session")
    if user_sessions.get(cookie):
        return True

    if username is None or password is None:
        return False

    pwd = user_pwd.get(username, None)
    if pwd == password:
        cookie = uuid.uuid4().hex
        user_sessions[cookie] = username
        response.set_cookie("zillionare-auth-session", cookie)
        return True
    return False


def get_current_user():
    global user_sessions
    cookie = flask.request.cookies.get("zillionare-auth-session")

    return user_sessions.get(cookie, None)


def remove_current_user():
    global user_sessions
    cookie = flask.request.cookies.get("zillionare-auth-session")

    if cookie in user_sessions:
        del user_sessions[cookie]
