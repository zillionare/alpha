import unittest
from alpha.notebook import *
import omicron

from tests import init_test_env

class TestNotebook(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        init_test_env()
        await omicron.init()

    async def test_scan(self):
        def echo(code, name, bars, results, ft):
            print(name, bars[-1]["frame"])
            results.append((name, code, bars[-1]["frame"]))

        results = await scan(echo, 5, '1d', nstocks=1)

    def test_performance(self):
        # 天齐锂业 2021-11-9
        bars = np.array([(datetime.date(2021, 10, 27),  98.71, 100.77,  95.88,  97.8 , 53176349., 5.20893753e+09, 7.76),
       (datetime.date(2021, 10, 28),  97.32,  99.99,  90.85,  91.83, 92554120., 8.72289423e+09, 7.76),
       (datetime.date(2021, 10, 29),  92.7 ,  94.72,  91.9 ,  94.26, 51045515., 4.77486161e+09, 7.76),
       (datetime.date(2021, 11, 1),  93.29,  95.88,  88.95,  89.19, 78450901., 7.17030458e+09, 7.76),
       (datetime.date(2021, 11, 2),  90.78,  93.11,  89.81,  91.06, 69944612., 6.40148764e+09, 7.76),
       (datetime.date(2021, 11, 3),  92.99,  93.05,  88.65,  91.99, 51965519., 4.73467128e+09, 7.76),
       (datetime.date(2021, 11, 4),  92.51,  95.4 ,  90.56,  91.69, 71685515., 6.65031555e+09, 7.76),
       (datetime.date(2021, 11, 5),  93.01,  97.38,  92.69,  94.98, 96145021., 9.17006088e+09, 7.76),
       (datetime.date(2021, 11, 8),  95.2 , 104.48,  95.2 , 104.48, 99840134., 1.01206507e+10, 7.76),
       (datetime.date(2021, 11, 9), 108.12, 109.45, 105.55, 108.44, 89927252., 9.68160988e+09, 7.76)],
      dtype=[('frame', 'O'), ('open', '<f4'), ('high', '<f4'), ('low', '<f4'), ('close', '<f4'), ('volume', '<f8'), ('amount', '<f8'), ('factor', '<f4')])

        profits = performance(bars, [1,3,5], stop_loss = 0.08)
        np.testing.assert_array_almost_equal(profits, [0.09857165813446045, -0.09644407033920288, -0.06969910860061646, -0.09644407033920288, -0.09644407033920288],3)

        profits = performance(bars, [1,3,5,9], stop_loss = 0.2)
        print(profits)
        np.testing.assert_array_almost_equal(profits, [0.09857165813446045, -0.09644407033920288, -0.06969910860061646, -0.09644407033920288, -0.06807821989059448, 0.09857165813446045], 3)


