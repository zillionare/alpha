import datetime
import unittest
from telnetlib import SE

from omicron import tf

from alpha.utils.data.securities import Securities


class SecuritiesTest(unittest.TestCase):
    def test_sync(self):
        try:
            secs = Securities("/tmp/securities.h5")

            tf.service_degrade()
            secs.sync("2022-03-10", "2022-03-18")

            self.assertSetEqual(
                set(secs.store.attrs.get("synced")),
                set(
                    [
                        "2022-03-10",
                        "2022-03-11",
                        "2022-03-14",
                        "2022-03-15",
                        "2022-03-16",
                        "2022-03-17",
                        "2022-03-18",
                    ]
                ),
            )

            secs.sync("2022-03-10", "2022-03-21")
            self.assertSetEqual(set(secs.store.attrs.get("synced"))), set(
                [
                    "2022-03-10",
                    "2022-03-11",
                    "2022-03-14",
                    "2022-03-15",
                    "2022-03-16",
                    "2022-03-17",
                    "2022-03-18",
                    "2022-03-19",
                    "2022-03-20",
                    "2022-03-21",
                ]
            )
        finally:
            secs.close()

    def test_query(self):
        secs = Securities("/tmp/securities.h5")

        tf.service_degrade()
        secs.sync("2022-03-10", "2022-03-18")

        now = datetime.date(2022, 3, 10)

        codes = secs.query(now).types(["stock"]).codes
        self.assertTrue(len(codes) > 4000)
        self.assertIn("000001.XSHE", codes)

        codes = secs.query(now).types(["index"]).codes
        self.assertTrue(len(codes) > 500)
        self.assertIn("000001.XSHG", codes)

        codes = (
            secs.query(now)
            .types(["stock"])
            .exclude_cyb()
            .exclude_exit(now)
            .exclude_kcb()
            .exclude_st()
            .name_like("银行")
            .codes
        )

        self.assertIn("002807.XSHE", codes)

        codes = secs.query(now).types(["stock"]).only_cyb().codes
        self.assertIn("300001.XSHE", codes)

        now = datetime.date(2022, 2, 1)
        codes = secs.query(now).types(["stock"]).only_cyb().codes
        self.assertEqual(len(codes), 0)

        now = datetime.date(2022, 3, 28)
        codes = secs.query(now).types(["stock"]).only_cyb().codes
        self.assertTrue(len(codes) > 300)
