import asyncio
import logging
import os
import platform
import smtplib
import tempfile
from email.message import EmailMessage

import aiohttp
import cfg4py
from IPython.display import Audio, display

from alpha.config import get_config_dir

cfg = cfg4py.init(get_config_dir())
logger = logging.getLogger(__name__)


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


# def init_tts():
#     import pyttsx3

#     _tts = pyttsx3.init()

#     if "macOS" in platform.platform():
#         voices = _tts.setProperty("voice", "com.apple.speech.synthesis.voice.mei-jia")
#     else:
#         _tts.setProperty("voice", "zh")

#     return _tts


# def say(text):
#     global _tts
#     _tts.say(text)
#     _tts.runAndWait()


async def text_to_speech(text):
    file = tempfile.mktemp(dir="/tmp/alpha/audio/", suffix=".wav")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{cfg.alpha.tts_server}text={text}") as resp:
            if resp.status == 200:
                with open(file, "wb") as f:
                    f.write(await resp.read())
                    return file

    return None


async def nb_say(text):
    file = await text_to_speech(text)

    display(_InvisibleAudio(filename=file, autoplay=True))


class _InvisibleAudio(Audio):
    """
    An invisible (`display: none`) `Audio` element which removes itself when finished playing.
    Taken from https://stackoverflow.com/a/50648266.
    """

    def _repr_html_(self) -> str:
        audio = super()._repr_html_()
        audio = audio.replace(
            "<audio", '<audio onended="this.parentNode.removeChild(this)"'
        )
        return f'<div style="display:none">{audio}</div>'


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
