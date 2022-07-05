# Usage

To use zillionare alpha in a project

```python
    """the main module"""
    from alpha import app, main

    import cfg4py
    import omicron
    from alpha import app, main
    from alpha.core.executor import create_process_pool
    from alpha.core.redislog import RedisLogReceiver
    from pyemit import emit
    from h2o_wave import Q
    from alpha.web.pages.research import research_view

    from zoo.config import get_config_dir


    async def on_client_connected(q: Q) -> None:
        q.user.theme = q.user.theme or "ember"

        # If no active hash present, render research.
        if q.args["#"] is None:
            await research_view(q)


    async def on_startup():
        cfg = cfg4py.init(get_config_dir())
        await omicron.init()
        await emit.start(emit.Engine.REDIS, dsn=cfg.redis.dsn)
        await RedisLogReceiver.listen()
        await create_process_pool(1)


    async def on_shutdown():
        await omicron.close()


    app.start(
        "/",
        on_app_startup=on_startup,
        on_app_shutdown=on_shutdown,
        on_client_connected=on_client_connected,
    )

```
