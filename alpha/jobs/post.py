"""生成jieyu.ai上的投资贴并发布"""
import os


class Post:
    def __init__(self):
        home = "~/workspace/zillionare/docs/investement/"
        self.home = os.path.expanduser(home)
