import os
import smtplib
from email.message import EmailMessage
import cfg4py
from alpha.config import get_config_dir
import pyttsx3

cfg = cfg4py.init(get_config_dir())


def send_html_email(
    subject: str,
    html_content: str,
    to_addrs: str = None,
    from_addrs: str = None,
    plain_text: str = None,
):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addrs or cfg.notify.mail_from
    msg["To"] = to_addrs or cfg.notify.mail_to
    msg.set_content(plain_text or "")

    msg.add_alternative(html_content, subtype="html")
    with smtplib.SMTP(cfg.notify.mail_server) as server:
        server.login(cfg.notify.mail_from, os.getenv("MAIL_PASSWORD"))
        server.send_message(msg)


def send_mail(subject: str, content: str, from_addrs: str = None, to_addrs: str = None):
    from_addrs = from_addrs or cfg.notify.mail_from
    to_addrs = to_addrs or cfg.notify.mail_to

    msg = EmailMessage()

    msg["Subject"] = subject
    msg["From"] = from_addrs or cfg.notify.mail_from
    msg["To"] = to_addrs or cfg.notify.mail_to
    msg.set_content(content)

    with smtplib.SMTP(cfg.notify.mail_server) as server:
        server.login(from_addrs or cfg.notify.mail_from, os.getenv("MAIL_PASSWORD"))
        server.send_message(msg)


def init_tts():
    _tts = pyttsx3.init()

    voices = _tts.setProperty("voice", "com.apple.speech.synthesis.voice.mei-jia")

    return _tts


def say(text):
    global _tts
    _tts.say(text)
    _tts.runAndWait()


_tts = init_tts()

if __name__ == "__main__":
    send_html_email(
        "this is a test",
        """\
<html>
  <head></head>
  <body>
    <h1>Salut!</h1>
    <p>Cela ressemble à un excellent
        <a href="http://www.yummly.com/recipe/Roasted-Asparagus-Epicurious-203718">
            recipie
        </a> déjeuner.
    </p>
  </body>
</html>""",
    )
