#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Aaron-Yang [code@jieyu.ai]
Contributors: 

"""
import datetime
import functools
import json
import logging
from json import JSONEncoder

from arrow import Arrow
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from omicron.models.securities import Securities
from omicron.models.security import Security
from sanic import response

from alpha.core.monitors import mm
from alpha.plots import create_plot

logger = logging.getLogger(__name__)


class MyJsonDumper(JSONEncoder):
    # Override the default method
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        if isinstance(obj, Arrow):
            return obj.datetime.isoformat()

        return json.JSONEncoder.default(self, obj)


async def get_stock_pool(request):
    args = request.args
    frames = int(args.get('frames')[0])
    frame_types = args.getlist("frame_types")

    if frame_types:
        frame_types = [FrameType(frame_type) for frame_type in frame_types]
    else:
        frame_types = tf.day_level_frames
        frame_types.extend(tf.minute_level_frames)

    plots = args.getlist('plots') or ['momentum']

    results = []
    for plot_name in plots:
        plot = create_plot(plot_name)
        results.append(await plot.list_stock_pool(frames, frame_types))

    return response.json(body=results)


async def plot_command_handler(request, cmd):
    plot_name = request.json.get("plot")
    params = request.json
    del params['plot']

    try:
        plot = create_plot(plot_name)
        func = getattr(plot, cmd)
        results = await func(**params)
        return response.json(results, status=200)
    except Exception as e:
        logger.exception(e)
        return response.json(e, status=500)


async def list_monitors(request):
    params = request.args
    code = params.get('code')
    frame_type = params.get('frame_type')
    plot = params.get('plot')
    flag = params.get('falg')

    try:
        monitors = await mm.list_monitors(code=code, frame_type=frame_type,
                                          plot=plot, flag=flag)

        result = {}
        for job_name, name, params, trigger in monitors:
            plot = create_plot(name)

            rows = result.get(plot.display_name, [])

            row = plot.translate_monitor(job_name, params, trigger)
            if row:
                rows.append(row)
                result[plot.display_name] = rows

        return response.json(result, status=200)
    except Exception as e:
        logger.exception(e)
        return response.text(e, status=500)


async def add_monitor(request):
    """
    supported cmd:
        add, remove,list
    Args:
        request:

    Returns:

    """
    params = request.json
    code = params.get("code")
    plot = params.get('plot')

    if all([code is None, plot is None]):
        return response.text("必须指定要监控的股票代码", status=401)

    try:
        del params['plot']

        await mm.add_monitor(plot, **params)
        display_name = Security(code).display_name
        return response.text(f"{display_name}已加入{plot}监控", status=200)
    except Exception as e:
        logger.exception(e)
        return response.text(e, status=500)


async def remove_monitor(request):
    params = request.json

    key = params.get("key")
    code = params.get("code")
    plot = params.get("plot")
    frame_type = params.get("frame_type")
    flag = params.get("flag")
    _remove_all = params.get("all")

    if _remove_all or any((key, code, plot)):
        try:
            removed = await mm.remove(key, plot=plot, code=code,
                                      frame_type=frame_type, flag=flag,
                                      remove_all=_remove_all)
            return response.json(removed, status=200)
        except Exception as e:
            return response.text(e, status=500)
    else:
        return response.json("code, plot are required", status=401)


async def fuzzy_match(request):
    query = request.args.get('query')
    results = Securities().fuzzy_match(query)

    dumps = functools.partial(json.dumps, cls=MyJsonDumper)
    return response.json(results, dumps=dumps)
