#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import logging
from typing import Tuple

import aiohttp

logger = logging.getLogger(__name__)

# -*- coding:utf-8 -*-
#
#   author: iflytek
#
#  本demo测试时运行的环境为：Windows + Python3.7
#  本demo测试成功运行时所安装的第三方库及其版本如下：
#   cffi==1.12.3
#   gevent==1.4.0
#   greenlet==0.4.15
#   pycparser==2.19
#   six==1.12.0
#   websocket==0.2.1
#   websocket-client==0.56.0
#   合成小语种需要传输小语种文本、使用小语种发音人vcn、tte=unicode以及修改文本编码方式
#  错误码链接：https://www.xfyun.cn/document/error-code （code返回错误码时必看）
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import os

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识


class ItekError(BaseException):
    pass


class ItekClient:
    APPID = '5ce6513b'
    APIKey = 'eeba7e937236a0a48cfa8f83badcfafc'
    APISecret = '7c2a6599d262cbddaa909f77aa536fbb'

    # 公共参数(common)
    common_args = {"app_id": APPID}
    # 业务参数(business)，更多个性化参数可在官网查看
    business_args = {"aue": "lame", "auf": "audio/L16;rate=16000", "sfl": 1,
                     "vcn": "xiaoyan", "tte": "utf8"}

    # 使用小语种须使用以下方式，此处的unicode指的是 utf16小端的编码方式，即"UTF-16LE"”
    # self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-16')), "UTF8")}
    def __init__(self, save_to: str = ""):
        self.save_to = save_to or "/tmp/"
        if not os.path.exists(save_to):
            os.makedirs(self.save_to, exist_ok=True)

    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'),
                                 signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(
                encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date":          date,
            "host":          "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url

    def make_msg(self, text: str) -> Tuple:
        url = self.create_url()

        data = {"status": 2,
                "text":   str(base64.b64encode(text.encode('utf-8')), "UTF8")}
        return url, json.dumps({"common":   self.common_args,
                                "business": self.business_args,
                                "data":     data,
                                })

    async def tts(self, text: str, filename: str):
        url, data = self.make_msg(text)
        path = os.path.join(self.save_to, filename)
        if os.path.exists(path):
            os.remove(path)

        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(url, ssl=False, timeout=20) as ws:
                await ws.send_str(data)
                async for message in ws:
                    try:
                        message = json.loads(message.data)
                        code = message["code"]
                        sid = message["sid"]
                        audio = message["data"]["audio"]
                        status = message["data"]["status"]

                        if status == 2:
                            await ws.close()
                        if code != 0:
                            err = message["message"]
                            logger.warning("sid:%s call error: %s code is :%s", sid, err,
                                           code)
                            raise ItekError(err)
                        else:
                            with open(path, 'ab') as f:
                                audio = base64.b64decode(audio)
                                f.write(audio)
                    except Exception as e:
                        logger.warning("received msg, but failed to parse: %s", text)
                        logger.exception(e)
                        raise ItekError("failed to parse response")

            return path
