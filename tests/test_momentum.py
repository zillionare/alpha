import functools
import unittest

import arrow
from alpha.core.enums import Events
from omicron.core.lang import async_run
from omicron.core.timeframe import tf
from omicron.core.types import FrameType
from pyemit import emit

from alpha.core.monitors import monitor
from alpha.plots.momentum import Momentum
from tests.base import AbstractTestCase


class MyTestCase(AbstractTestCase):
    @async_run
    async def test_momentum_scan(self):
        plot = Momentum()
        # end = arrow.get('2020-8-26').date()
        monitor.init()
        # await plot.scan(FrameType.DAY, end, codes=['300606.XSHE'])
        #
        # end = arrow.get('2020-8-28 15:00').datetime
        # await plot.scan(FrameType.MIN30, end, codes=['300095.XSHE'])

        await plot.scan(FrameType.MIN30)

    @async_run
    async def test_momentum_visualize(self):
        plot = Momentum()

        end = arrow.get('2020-8-28')
        frame_type = FrameType.DAY
        await plot.visualize(
                ['300129.XSHE', '300859.XSHE', '300729.XSHE', '300612.XSHE',
                 '300589.XSHE'], end, frame_type)

    @async_run
    async def test_test_signal(self):
        plot = Momentum()

        results = {}
        async def on_trade_signal(results, msg):
            results[msg.get('fire_on')] = {
                "flag": msg.get('flag'),
                "frame_type": msg.get('frame_type')
            }

        emit.register(Events.sig_trade, functools.partial(on_trade_signal, results))
        for frame in tf.get_frames(arrow.get('2020-8-24 11:00'),
                                   arrow.get('2020-8-28 15:00'),
                                   FrameType.MIN30):
            frame = tf.int2time(frame)
            await plot.evaluate('000001.XSHG', '30m', frame, flag='both')

        self.assertDictEqual({
            202008261000: {"flag": "short", "frame_type": '30m'},
            202008271130: {"flag": "long", "frame_type": '30m'}
        }, results)


if __name__ == '__main__':
    unittest.main()
