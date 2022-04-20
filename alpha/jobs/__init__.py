import asyncio
from threading import Thread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from .stockpooling import ting

def thread_main(loop):
    asyncio.set_event_loop(loop)

    scheduler = AsyncIOScheduler(loop=loop)
    scheduler.add_job(ting, "interval", seconds=5)
    scheduler.start()

    loop.run_forever()

def start_background_tasks():
    loop = asyncio.new_event_loop()

    t = Thread(target=thread_main, args=(loop,))
    t.start()
